import streamlit as st
import os
import tarfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from glob import glob
from astropy.io import fits
import astropy.units as u
import astropy.coordinates as coord
from dust_extinction.parameter_averages import CCM89
from astroquery.irsa_dust import IrsaDust

# pPXF 코어 모듈 임포트
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

# ==============================================================================
# [페이지 초기 설정 및 세션 상태(메모리) 초기화]
# ==============================================================================
st.set_page_config(page_title="AstroFit", layout="wide")

# 리포트용 그래픽 자산 저장 경로 설정
IMAGE_DIR = "./report_assets"
os.makedirs(IMAGE_DIR, exist_ok=True)

if "metadata" not in st.session_state:
    st.session_state.metadata = {"obj_name": "NGC 5548", "ra": 18.87685, "dec": -0.86098, "obj_type": "Seyfert Galaxy"}

if "config" not in st.session_state:
    st.session_state.config = {"fits_file": None, "emiles_file": None, "manual_Av": "0.0700", "Rv": 3.1}

if "pipeline_data_stream" not in st.session_state:
    st.session_state.pipeline_data_stream = {
        "templates": [],
        "wave_obs": None, "flux_obs": None, "sigma_obs": None,
        "wave_rest": None, "flux_dereddened": None, "sigma_dereddened": None,
        "z_calculated": None, "final_Av": None,
        "velscale": None, "sigma_stars": None, "sigma_err": None,
        "log_M_bh": None, "log_M_bh_err": None, "M_bh": None,
        "M_bh_lower": None, "M_bh_upper": None,
        "str_mass_center": None, "str_mass_range": None,
        "stellar_continuum": None, "gas_fit": None, "pp_object": None,
        "saved_plots": {},   # 리포트 생성 엔진용 이미지 주소록
        "is_ready": False,
        "is_ppxf_ready": False
    }

OBS_WAVE_RANGE  = (4300, 9500)
REST_WAVE_RANGE = (4000, 9000)
ZOOM_WAVE_RANGE = (6400, 6600)

# ==============================================================================
# [HUMAN-READABLE UTILITY] 태양질량 직관적 단위 변환 함수
# ==============================================================================
def to_korean_shares(value):
    """숫자를 'X억 X,XXX만' 형태의 직관적인 한국어 배수로 변환합니다."""
    if value >= 1e8:
        eok = int(value // 1e8)
        man = int((value % 1e8) // 1e4)
        return f"{eok}억 {man:,}만" if man > 0 else f"{eok}억"
    elif value >= 1e4:
        man = int(value // 1e4)
        return f"{man:,}만"
    else:
        return f"{value:,.0f}"

# ==============================================================================
# [CORE ALGORITHM SCIENTIFIC MODULES] 과학 연산 모듈
# ==============================================================================
def setup_templates(uploaded_tar, extract_path="./temp_emiles"):
    if uploaded_tar is None: return []
    shutil.rmtree(extract_path, ignore_errors=True)
    os.makedirs(extract_path, exist_ok=True)
    with tarfile.open(fileobj=uploaded_tar, mode="r:gz") as tar:
        tar.extractall(path=extract_path)
    return sorted(glob(f"{extract_path}/**/*.fits", recursive=True))

def load_and_process_spectrum(uploaded_fits, manual_Av_str, Rv=3.1):
    if uploaded_fits is None: raise FileNotFoundError("SDSS FITS 파일이 없습니다.")
    with fits.open(uploaded_fits) as hdul:
        coadd = hdul[1].data
        specobj = hdul[2].data
        header = hdul[0].header
        flux_obs = coadd['flux']
        wave_obs = 10**coadd['loglam']
        ivar = coadd['ivar']
        z = specobj['Z'][0]
        sigma_obs = np.zeros_like(flux_obs)
        good_pixels = ivar > 0
        sigma_obs[good_pixels] = 1.0 / np.sqrt(ivar[good_pixels])

        try:
            if manual_Av_str.upper() != 'NONE' and manual_Av_str.strip() != '':
                Av = float(manual_Av_str)
            else: raise ValueError
        except ValueError:
            ra, dec = header['RA'], header['DEC']
            try:
                target_coord = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
                dust_table = IrsaDust.get_query_table(target_coord, section='ebv')
                Av = Rv * dust_table['extSandF'][0]
            except: Av = 0.0700

        ext_model = CCM89(Rv=Rv)
        transmission = ext_model.extinguish(wave_obs * u.AA, Av=Av)
        flux_dereddened = flux_obs / transmission
        sigma_dereddened = sigma_obs / transmission
        wave_rest = wave_obs / (1 + z)
        return wave_obs, flux_obs, sigma_obs, wave_rest, flux_dereddened, sigma_dereddened, z, Av

# ==============================================================================
# [VISUALIZATION PLOTS] 차트 생성 및 디스크 자동 저장 컴포넌트
# ==============================================================================
def plot_observed_frame(wave_obs, flux_obs, sigma_obs, wave_range, save_path, step=10):
    mask = (wave_obs >= wave_range[0]) & (wave_obs <= wave_range[1])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(wave_obs[mask], flux_obs[mask], color='darkgray', lw=0.8, label="Raw Observed Spectrum")
    ax.errorbar(wave_obs[mask][::step], flux_obs[mask][::step], yerr=sigma_obs[mask][::step],
                 fmt='none', ecolor='salmon', elinewidth=0.5, capsize=1, alpha=0.4, label=r"1$\sigma$ Noise")
    ax.set_xlabel("Observed Wavelength (Å)", fontsize=12)
    ax.set_ylabel("Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=12)
    ax.set_title("1. Pure Original Spectrum (Observed Frame, Pre-corrections)", fontsize=12, fontweight='bold')
    ax.set_xlim(wave_range)
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_rest_frame_original(wave_rest, flux_obs, sigma_obs, wave_range, save_path, step=10):
    mask = (wave_rest >= wave_range[0]) & (wave_rest <= wave_range[1])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(wave_rest[mask], flux_obs[mask], color='black', lw=0.8, label="Redshift-Corrected Spectrum")
    ax.errorbar(wave_rest[mask][::step], flux_obs[mask][::step], yerr=sigma_obs[mask][::step],
                 fmt='none', ecolor='red', elinewidth=0.5, capsize=1, alpha=0.5, label=r"1$\sigma$ Noise")
    ax.set_xlabel("Rest Wavelength (Å)", fontsize=12)
    ax.set_ylabel("Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=12)
    ax.set_title("2. Rest-Frame Spectrum (Wavelength Shifted, Before Dust Correction)", fontsize=12, fontweight='bold')
    ax.set_xlim(wave_range)
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_dust_correction_comparison(wave_rest, flux_obs, flux_corr, wave_range, save_path):
    mask = (wave_rest >= wave_range[0]) & (wave_rest <= wave_range[1])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(wave_rest[mask], flux_obs[mask], color='black', lw=0.8, label='Before Dust Correction')
    ax.plot(wave_rest[mask], flux_corr[mask], color='firebrick', lw=0.8, label='CCM89 Corrected (Final)')
    ax.set_xlabel("Rest Wavelength (Å)", fontsize=12)
    ax.set_ylabel("Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=12)
    ax.set_title("3. Galactic Dust Extinction Correction Comparison", fontsize=12, fontweight='bold')
    ax.set_xlim(wave_range)
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_emission_lines_zoom(wave_rest, flux_corr, wave_range, save_path):
    mask = (wave_rest >= wave_range[0]) & (wave_rest <= wave_range[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wave_rest[mask], flux_corr[mask], color='black', lw=1.0, label='Dereddened Flux')
    lines = {r'[N II] $\lambda$6548': 6548.05, r'H$\alpha$': 6562.80, r'[N II] $\lambda$6583': 6583.45}
    colors = ['green', 'red', 'green']
    for (label, wave), color in zip(lines.items(), colors):
        ax.axvline(wave, color=color, ls='--', lw=1.2, label=label)
    ax.set_xlabel("Rest Wavelength (Å)", fontsize=12)
    ax.set_ylabel("Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=12)
    ax.set_title("4. Zoom-in: Key Emission Lines for Modeling Readiness", fontsize=12, fontweight='bold')
    ax.set_xlim(wave_range)
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(frameon=False, loc='upper left')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_ppxf_fit(wave_rest, galaxy_flux, bestfit, goodpixels, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    clean_idx = goodpixels

    ax1.plot(wave_rest[clean_idx], galaxy_flux[clean_idx], color='black', lw=0.8, label='Observed (Dereddened)')
    ax1.plot(wave_rest[clean_idx], bestfit[clean_idx], color='red', lw=1.2, label='pPXF Stellar+Gas Fit')
    ax1.set_ylabel("Relative Flux ($f_\\lambda$)", fontsize=11)
    ax1.set_title("pPXF Perfect Fit (Stellar Continuum + AGN Emission Lines)", fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=False)
    ax1.tick_params(direction='in', top=True, right=True)

    ymin = max(-5, np.percentile(galaxy_flux[clean_idx], 0.5) - 2)
    ymax = np.percentile(galaxy_flux[clean_idx], 99.8) * 1.2
    ax1.set_ylim(ymin, ymax)

    residuals = galaxy_flux[clean_idx] - bestfit[clean_idx]
    ax2.plot(wave_rest[clean_idx], residuals, 'd', color='limegreen', markersize=2.5, label='Residuals', alpha=0.8)
    ax2.axhline(0, color='gray', linestyle='--', lw=0.8)
    ax2.set_xlabel(r"$\lambda_{\rm rest}$ (Å)", fontsize=11)
    ax2.set_ylabel("Residuals", fontsize=11)
    ax2.legend(loc='upper right', frameon=False)
    ax2.tick_params(direction='in', top=True, right=True)
    ax2.set_ylim(-np.percentile(np.abs(residuals), 95)*3, np.percentile(np.abs(residuals), 95)*3)
    ax1.set_xlim(3700.0, 10500.0)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_spectral_decomposition(wave_rest, original_flux, bestfit, stellar_continuum, residual_dec, mask_dec, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    wave_min, wave_max = 3800, 7000

    ax1.plot(wave_rest[mask_dec], original_flux[mask_dec], color='black', lw=0.8, label='Original (Dereddened)')
    ax1.plot(wave_rest[mask_dec], bestfit[mask_dec], color='firebrick', lw=0.8, label='Total pPXF Fit')
    ax1.plot(wave_rest[mask_dec], stellar_continuum[mask_dec], color='navy', lw=0.8, linestyle='--', label='Stellar Continuum')
    ax1.set_xlim(wave_min, wave_max)
    ax1.set_ylabel("Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=12)
    ax1.set_title('pPXF Spectral Decomposition & Verification', fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylim(-10, np.percentile(original_flux[mask_dec], 99.8) * 1.3)
    ax1.tick_params(direction='in', top=True, right=True)
    ax1.legend(frameon=False, fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.2, linestyle=':')

    ax2.plot(wave_rest[mask_dec], residual_dec[mask_dec], color='gray', lw=0.8, label='Residuals')
    ax2.axhline(0, color='black', linestyle=':', alpha=0.6, lw=0.8)
    ax2.set_xlabel("Rest wavelength (Å)", fontsize=12)
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.set_ylim(-np.percentile(np.abs(residual_dec[mask_dec]), 95)*3, np.percentile(np.abs(residual_dec[mask_dec]), 95)*3)
    ax2.tick_params(direction='in', top=True, right=True)
    ax2.legend(frameon=False, fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.2, linestyle=':')

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

# ==============================================================================
# [MENU NAVIGATION & UI CONTROL PANEL]
# ==============================================================================
st.sidebar.title("AstroFit 시스템")
menu = st.sidebar.radio("이동할 페이지를 선택하세요:", ["1. 마스터 제어판 (Control Panel)", "2. pPXF 연속광 공제 설명", "3. Hβ 성분 분해 설명", "4. 비리얼 블랙홀 질량 계산"])

if menu == "1. 마스터 제어판 (Control Panel)":
    st.subheader("Spectrum Analysis Report Master Control Panel")
    
    st.markdown("**외부 데이터베이스 및 템플릿 다운로드 빠른 링크**")
    link_col1, link_col2, link_col3 = st.columns(3)
    with link_col1:
        st.link_button("SDSS DR19 CAS 바로가기", "https://cas.sdss.org/dr19", use_container_width=True)
    with link_col2:
        st.link_button("NASA IRSA Dust 조회", "https://irsa.ipac.caltech.edu/applications/DUST/", use_container_width=True)
    with link_col3:
        st.link_button("E-MILES 템플릿 다운로드", "https://cloud.iac.es/index.php/s/aYECNyEQfqgYwt4?dir=/E-MILES", use_container_width=True)
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### [A] 보고서 출력 정보")
        obj_name = st.text_input("천체 이름:", value=st.session_state.metadata["obj_name"])
        ra = st.number_input("적경 (RA):", value=st.session_state.metadata["ra"], format="%.5f")
        dec = st.number_input("적위 (DEC):", value=st.session_state.metadata["dec"], format="%.5f")
        type_options = ['Seyfert Galaxy', 'Seyfert 1', 'Seyfert 2', 'QSO', 'HII Galaxy', 'LINER', 'Blazar']
        obj_type = st.selectbox("천체 유형:", options=type_options, index=type_options.index(st.session_state.metadata["obj_type"]))

    with col2:
        st.markdown("### [B] 백엔드 데이터 실측 적용")
        fits_file = st.file_uploader("SDSS FITS 파일 선택 (.fits)", type=["fits", "fits.gz"])
        emiles_file = st.file_uploader("E-MILES 템플릿 파일 선택 (.tar.gz)", type=["gz", "tar.gz"])
        manual_Av = st.text_input("성간소광량 Av (or None):", value=st.session_state.config["manual_Av"])
        Rv = st.number_input("Rv 상수 (기본 3.1):", value=st.session_state.config["Rv"], format="%.1f")

    if st.button("제어판 데이터 시스템 동기화 (Apply)", use_container_width=True):
        st.session_state.metadata.update({"obj_name": obj_name.strip(), "ra": ra, "dec": dec, "obj_type": obj_type})
        st.session_state.config.update({"fits_file": fits_file, "emiles_file": emiles_file, "manual_Av": manual_Av, "Rv": Rv})
        st.success("제어판 파라미터 세션 저장 완료")

    st.write("---")
    st.markdown("### 1단계: 과학 연산 파이프라인 가동")
    
    if st.button("적색편이 및 데이터 보정 파이프라인 가동", type="primary", use_container_width=True):
        if st.session_state.config["fits_file"] is None or st.session_state.config["emiles_file"] is None:
            st.error("에러: FITS 파일과 템플릿 파일을 업로드한 후 동기화(Apply)를 먼저 진행해주세요.")
        else:
            with st.spinner("1단계 전처리 및 성간 소광 자동 연산 수행 중..."):
                try:
                    templates_list = setup_templates(st.session_state.config["emiles_file"])
                    w_obs, f_obs, s_obs, w_rest, f_corr, s_corr, calc_z, final_av = load_and_process_spectrum(
                        st.session_state.config["fits_file"], st.session_state.config["manual_Av"], st.session_state.config["Rv"]
                    )
                    
                    path_p1 = os.path.join(IMAGE_DIR, "01_observed_frame.png")
                    path_p2 = os.path.join(IMAGE_DIR, "02_rest_frame.png")
                    path_p3 = os.path.join(IMAGE_DIR, "03_dust_comparison.png")
                    path_p4 = os.path.join(IMAGE_DIR, "04_emission_lines_zoom.png")
                    
                    st.session_state.pipeline_data_stream.update({
                        "templates": templates_list, "wave_obs": w_obs, "flux_obs": f_obs, "sigma_obs": s_obs,
                        "wave_rest": w_rest, "flux_dereddened": f_corr, "sigma_dereddened": s_corr,
                        "z_calculated": calc_z, "final_Av": final_av,
                        "saved_plots": {
                            "observed_frame": path_p1,
                            "rest_frame": path_p2,
                            "dust_comparison": path_p3,
                            "emission_zoom": path_p4
                        },
                        "is_ready": True
                    })
                    
                    st.success(f"전처리 완료: z = {calc_z:.6f} | Av = {final_av:.4f} mag (차트 백업 완료)")
                    
                    st.markdown("#### 실시간 물리 데이터 검수 그래프")
                    st.pyplot(plot_observed_frame(w_obs, f_obs, s_obs, OBS_WAVE_RANGE, path_p1))
                    st.pyplot(plot_rest_frame_original(w_rest, f_obs, s_corr, REST_WAVE_RANGE, path_p2))
                    st.pyplot(plot_dust_correction_comparison(w_rest, f_obs, f_corr, REST_WAVE_RANGE, path_p3))
                    st.pyplot(plot_emission_lines_zoom(w_rest, f_corr, ZOOM_WAVE_RANGE, path_p4))
                    
                except Exception as e:
                    st.error(f"파이프라인 연산 중 치명적 오류 발생: {e}")

    # ==============================================================================
    # 2단계: pPXF 분석 및 M-sigma 관계식 기반 블랙홀 질량 산출 파이프라인
    # ==============================================================================
    st.write("---")
    st.markdown("### 2단계: 풀 스펙트럼 피팅(pPXF) 및 블랙홀 질량 산출 파이프라인 가동")
    
    if st.button("pPXF 최적화 및 블랙홀 질량 계산 파이프라인 가동", type="primary", use_container_width=True):
        if not st.session_state.pipeline_data_stream.get("is_ready", False):
            st.error("실행 실패: 1단계 데이터 보정 파이프라인이 아직 가동되지 않았습니다. 상단의 1단계 버튼을 먼저 실행해주세요.")
        else:
            with st.spinner("pPXF 종합 풀 스펙트럼 피팅 비선형 최적화 및 오차 전파 연산 수행 중..."):
                try:
                    stream = st.session_state.pipeline_data_stream
                    files = stream["templates"]
                    galaxy_wave_obs = stream["wave_obs"]
                    galaxy_flux = stream["flux_dereddened"]
                    galaxy_noise = stream["sigma_dereddened"]
                    redshift = stream["z_calculated"]

                    path_p5 = os.path.join(IMAGE_DIR, "05_ppxf_perfect_fit.png")
                    path_p6 = os.path.join(IMAGE_DIR, "06_spectral_decomposition.png")

                    # 데이터 격자 정렬 및 로그 레빈
                    c = 299792.458
                    log_lam_gal = np.log(galaxy_wave_obs)
                    velscale = c * (log_lam_gal[-1] - log_lam_gal[0]) / (len(galaxy_wave_obs) - 1)

                    first_file = files[0]
                    with fits.open(first_file, mode='readonly') as hdu:
                        h = hdu[0].header
                        naxis1 = h.get('NAXIS1', len(hdu[0].data))
                        crval1 = h.get('CRVAL1')
                        cdelt1 = h.get('CDELT1')
                        lam_temp = crval1 + cdelt1 * np.arange(naxis1)
                        lam_range_temp = [lam_temp[0], lam_temp[-1]]

                    star_templates_list = []
                    for path in files:
                        with fits.open(path, mode='readonly') as hdu:
                            flux_temp = hdu[0].data
                            flux_log_temp, log_lam_temp, _ = util.log_rebin(lam_range_temp, flux_temp, velscale=velscale)
                            star_templates_list.append(flux_log_temp)
                    star_templates = np.column_stack(star_templates_list)

                    interpolator = interp1d(log_lam_temp, star_templates, axis=0, bounds_error=False, fill_value=0.0)
                    star_templates_aligned = interpolator(log_lam_gal)

                    fwhm_gal = 2.4
                    gas_templates, gas_names, line_wave = util.emission_lines(
                        log_lam_gal, [galaxy_wave_obs[0], galaxy_wave_obs[-1]], fwhm_gal
                    )

                    templates = np.column_stack([star_templates_aligned, gas_templates])
                    component = [0] * star_templates_aligned.shape[1] + [1] * gas_templates.shape[1]

                    # pPXF 피팅 구동
                    vel_init = c * np.log(1.0 + redshift)
                    start = [[vel_init, 150.0], [vel_init, 120.0]]
                    moments = [2, 2]

                    wave_rest = galaxy_wave_obs / (1.0 + redshift)
                    wave_limit = 3800.0
                    goodpixels = np.where((wave_rest > wave_limit) & (galaxy_flux > -1000))[0]

                    pp = ppxf(templates, galaxy_flux, galaxy_noise, velscale, start,
                              goodpixels=goodpixels, plot=False, degree=4, moments=moments, component=component)

                    # 오차 전파 및 블랙홀 질량 계산
                    sigma_stars = pp.sol[0][1]
                    try:
                        sigma_err = pp.error[0][1] if (hasattr(pp, 'error') and pp.error is not None) else 5.0
                    except:
                        sigma_err = 5.0

                    log_M_bh = 8.49 + 4.38 * np.log10(sigma_stars / 200.0)
                    M_bh_power = 10**log_M_bh

                    log_M_bh_err_meas = 4.38 * (sigma_err / (sigma_stars * np.log(10)))
                    intrinsic_scatter = 0.29
                    log_M_bh_err_total = np.sqrt(log_M_bh_err_meas**2 + intrinsic_scatter**2)

                    M_bh_lower = 10**(log_M_bh - log_M_bh_err_total)
                    M_bh_upper = 10**(log_M_bh + log_M_bh_err_total)

                    str_mass_center = to_korean_shares(M_bh_power)
                    str_mass_lower  = to_korean_shares(M_bh_lower)
                    str_mass_upper  = to_korean_shares(M_bh_upper)

                    # 성분 분해 결과 처리
                    n_stars = star_templates_aligned.shape[1]
                    n_gas = gas_templates.shape[1]
                    gas_fit = pp.matrix[:, n_stars:n_stars+n_gas] @ pp.weights[n_stars:n_stars+n_gas]
                    stellar_continuum = pp.bestfit - gas_fit
                    residual_dec = pp.galaxy - pp.bestfit
                    mask_dec = (wave_rest >= 3800) & (wave_rest <= 7000)

                    # 전역 데이터 업데이트
                    st.session_state.pipeline_data_stream.update({
                        "velscale": velscale, "sigma_stars": sigma_stars, "sigma_err": sigma_err,
                        "log_M_bh": log_M_bh, "log_M_bh_err": log_M_bh_err_total,
                        "M_bh": M_bh_power, "M_bh_lower": M_bh_lower, "M_bh_upper": M_bh_upper,
                        "str_mass_center": str_mass_center, "str_mass_range": f"{str_mass_lower} 배 ~ {str_mass_upper} 배",
                        "stellar_continuum": stellar_continuum, "gas_fit": gas_fit, "pp_object": pp,
                        "is_ppxf_ready": True
                    })
                    st.session_state.pipeline_data_stream["saved_plots"].update({
                        "ppxf_fit": path_p5,
                        "decomposition": path_p6
                    })

                    st.success(f"2단계 파이프라인 수렴 완료: 항성 속도분산 = {sigma_stars:.2f} km/s | 중심 블랙홀 질량 = 태양의 약 {str_mass_center} 배")

                    # 수치 리포트 대시보딩
                    st.markdown("#### AGN 블랙홀 질량 및 통계적 오차 산출 명세")
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric(label="항성 속도분산 측정치", value=f"{sigma_stars:.2f} ± {sigma_err:.2f} km/s")
                        st.text(f"물리 학술지 표기용 로그값: Log(M_BH/M_sun) = {log_M_bh:.2f} ± {log_M_bh_err_total:.2f}")
                    with metrics_col2:
                        st.metric(label="중심 블랙홀 질량 (대표값)", value=f"태양 질량의 {str_mass_center} 배")
                        st.text(f"1-σ 신뢰구간 범위: {str_mass_lower} 배 ~ {str_mass_upper} 배")

                    # 최적화 결과 시각화
                    st.markdown("#### pPXF 최적 모델 및 성분 분해 검수 그래프")
                    fig_fit = plot_ppxf_fit(wave_rest, galaxy_flux, pp.bestfit, goodpixels, path_p5)
                    st.pyplot(fig_fit)
                    plt.close(fig_fit)

                    fig_dec = plot_spectral_decomposition(wave_rest, pp.galaxy, pp.bestfit, stellar_continuum, residual_dec, mask_dec, path_p6)
                    st.pyplot(fig_dec)
                    plt.close(fig_dec)

                except Exception as e:
                    st.error(f"pPXF 최적화 파이프라인 연산 중 치명적 오류 발생: {e}")

elif menu == "2. pPXF 연속광 공제 설명":
    st.header("pPXF 항성 연속광 공제")
    st.write("---")
    if st.session_state.pipeline_data_stream["is_ppxf_ready"]:
        st.success(f"현재 로드된 천체 {st.session_state.metadata['obj_name']}의 pPXF 연산 데이터가 준비되어 있습니다.")
        st.write(f"항성 속도분산 고유 모델 값: {st.session_state.pipeline_data_stream['sigma_stars']:.2f} km/s")
    else:
        st.info("1번 제어판에서 1단계 및 2단계 파이프라인을 먼저 가동해 주세요.")

elif menu == "3. Hβ 성분 분해 설명":
    st.header("광폭 Hβ 방출선 성분 분해")
    st.write("---")

elif menu == "4. 비리얼 블랙홀 질량 계산":
    st.header("비리얼 정리 기반 블랙홀 질량 계산")
    st.write("---")
