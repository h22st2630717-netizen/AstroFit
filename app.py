import streamlit as st
import os
import tarfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.coordinates as coord
from glob import glob
from astropy.io import fits
from dust_extinction.parameter_averages import CCM89
from astroquery.irsa_dust import IrsaDust

# ==============================================================================
# [페이지 초기 설정 및 세션 상태(메모리) 초기화]
# ==============================================================================
st.set_page_config(page_title="AstroFit", page_icon="🌌", layout="wide")

# 리포트용 그래픽 자산 저장 경로 설정 (스트림릿 가상 서버 내부 디스크)
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
        "saved_plots": {},   # 🔥 리포트 생성 엔진이 참조할 이미지 주소록 백업용
        "is_ready": False
    }

OBS_WAVE_RANGE  = (4300, 9500)
REST_WAVE_RANGE = (4000, 9000)
ZOOM_WAVE_RANGE = (6400, 6600)

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
# [VISUALIZATION PLOTS] 렌더링 후 디스크 자동 저장 기능 복구 
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
    
    # 💾 기존 코랩 고해상도 이미지 파일 저장 로직 부활
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

# ==============================================================================
# [MENU NAVIGATION & UI CONTROL PANEL]
# ==============================================================================
st.sidebar.title("🌌 AstroFit 시스템")
menu = st.sidebar.radio("이동할 페이지를 선택하세요:", ["1. 마스터 제어판 (Control Panel)", "2. pPXF 연속광 공제 설명", "3. Hβ 성분 분해 설명", "4. 비리얼 블랙홀 질량 계산"])

if menu == "1. 마스터 제어판 (Control Panel)":
    st.subheader("⚙️ Spectrum Analysis Report Master Control Panel")
    
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

    if st.button("🔄 제어판 데이터 시스템 동기화 (Apply)", use_container_width=True):
        st.session_state.metadata.update({"obj_name": obj_name.strip(), "ra": ra, "dec": dec, "obj_type": obj_type})
        st.session_state.config.update({"fits_file": fits_file, "emiles_file": emiles_file, "manual_Av": manual_Av, "Rv": Rv})
        st.success("제어판 파라미터 세션 저장 완료!")

    st.write("---")
    st.markdown("### 🚀 과학 연산 파이프라인 가동")
    
    if st.button("▶️ 적색편이 및 데이터 보정 파이프라인 가동", type="primary", use_container_width=True):
        if st.session_state.config["fits_file"] is None or st.session_state.config["emiles_file"] is None:
            st.error("❌ 에러: FITS 파일과 템플릿 파일을 업로드한 후 동기화(Apply)를 먼저 진행해주세요.")
        else:
            with st.spinner("⏳ 1단계 전처리 및 성간 소광 자동 연산 수행 중..."):
                try:
                    templates_list = setup_templates(st.session_state.config["emiles_file"])
                    w_obs, f_obs, s_obs, w_rest, f_corr, s_corr, calc_z, final_av = load_and_process_spectrum(
                        st.session_state.config["fits_file"], st.session_state.config["manual_Av"], st.session_state.config["Rv"]
                    )
                    
                    # 💾 저장될 이미지 파일 경로 정의
                    path_p1 = os.path.join(IMAGE_DIR, "01_observed_frame.png")
                    path_p2 = os.path.join(IMAGE_DIR, "02_rest_frame.png")
                    path_p3 = os.path.join(IMAGE_DIR, "03_dust_comparison.png")
                    path_p4 = os.path.join(IMAGE_DIR, "04_emission_lines_zoom.png")
                    
                    # 💾 데이터 스트림에 파일 매핑 주소록(saved_plots) 저장!
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
                    
                    st.success(f"🎉 전처리 완료! z: {calc_z:.6f} | Av: {final_av:.4f} mag (차트 백업 완료)")
                    
                    # 실시간 렌더링 수행 (동시에 내부 디스크 물리 저장 작동)
                    st.markdown("#### 📉 실시간 물리 데이터 검수 그래프")
                    st.pyplot(plot_observed_frame(w_obs, f_obs, s_obs, OBS_WAVE_RANGE, path_p1))
                    st.pyplot(plot_rest_frame_original(w_rest, f_obs, s_corr, REST_WAVE_RANGE, path_p2))
                    st.pyplot(plot_dust_correction_comparison(w_rest, f_obs, f_corr, REST_WAVE_RANGE, path_p3))
                    st.pyplot(plot_emission_lines_zoom(w_rest, f_corr, ZOOM_WAVE_RANGE, path_p4))
                    
                except Exception as e:
                    st.error(f"파이프라인 연산 중 치명적 오류 발생: {e}")

elif menu == "2. pPXF 연속광 공제 설명":
    st.header("✨ pPXF 항성 연속광 공제")
    st.write("---")
    if st.session_state.pipeline_data_stream["is_ready"]:
        st.success(f"백업된 차트 주소록: {st.session_state.pipeline_data_stream['saved_plots']}")
