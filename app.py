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

if "metadata" not in st.session_state:
    st.session_state.metadata = {
        "obj_name": "NGC 5548",
        "ra": 18.87685,
        "dec": -0.86098,
        "obj_type": "Seyfert Galaxy"
    }

if "config" not in st.session_state:
    st.session_state.config = {
        "fits_file": None,
        "emiles_file": None,
        "manual_Av": "0.0700",
        "Rv": 3.1
    }

# 코랩의 pipeline_data_stream 전역 변수를 스트림릿 메모리에 바인딩
if "pipeline_data_stream" not in st.session_state:
    st.session_state.pipeline_data_stream = {
        "templates": [],
        "wave_obs": None, "flux_obs": None, "sigma_obs": None,
        "wave_rest": None, "flux_dereddened": None, "sigma_dereddened": None,
        "z_calculated": None, "final_Av": None,
        "is_ready": False
    }

OBS_WAVE_RANGE  = (4300, 9500)
REST_WAVE_RANGE = (4000, 9000)
ZOOM_WAVE_RANGE = (6400, 6600)

# ==============================================================================
# [CORE ALGORITHM SCIENTIFIC MODULES] 과학 연산 모듈 (웹 업로더 대응 변환)
# ==============================================================================
def setup_templates(uploaded_tar, extract_path="./temp_emiles"):
    """ 코랩 경로 대신 업로드된 tar.gz 파일 객체를 직접 읽어 압축을 풉니다 """
    if uploaded_tar is None:
        return []
    shutil.rmtree(extract_path, ignore_errors=True)
    os.makedirs(extract_path, exist_ok=True)
    
    # 가상 바이트 스트림 파일 읽기
    with tarfile.open(fileobj=uploaded_tar, mode="r:gz") as tar:
        tar.extractall(path=extract_path)
    return sorted(glob(f"{extract_path}/**/*.fits", recursive=True))

def load_and_process_spectrum(uploaded_fits, manual_Av_str, Rv=3.1):
    """ 코랩 경로 대신 업로드된 FITS 파일 객체를 직접 열어 물리 연산을 수행합니다 """
    if uploaded_fits is None:
        raise FileNotFoundError("지정하신 SDSS FITS 파일이 업로드되지 않았습니다.")

    # 주입된 파일 객체를 바로 FITS 렌더러에 전달
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

        # Av 값 수치 예외 처리
        try:
            if manual_Av_str.upper() != 'NONE' and manual_Av_str.strip() != '':
                Av = float(manual_Av_str)
                st.info(f" 💡 [소광 보정] 사용자가 지정한 고정값 적용: Av = {Av} mag")
            else:
                raise ValueError
        except ValueError:
            ra, dec = header['RA'], header['DEC']
            try:
                target_coord = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
                dust_table = IrsaDust.get_query_table(target_coord, section='ebv')
                Av = Rv * dust_table['extSandF'][0]
                st.info(f" 🛰️ [소광 보정] NASA IRSA 자동 조회 결과 적용: Av = {Av:.4f} mag")
            except:
                Av = 0.0700
                st.info(f" ⚠️ [소광 보정] NASA 조회 실패로 기본 백업값 적용: Av = {Av} mag")

        ext_model = CCM89(Rv=Rv)
        transmission = ext_model.extinguish(wave_obs * u.AA, Av=Av)

        flux_dereddened = flux_obs / transmission
        sigma_dereddened = sigma_obs / transmission
        wave_rest = wave_obs / (1 + z)

        return wave_obs, flux_obs, sigma_obs, wave_rest, flux_dereddened, sigma_dereddened, z, Av

# ==============================================================================
# [VISUALIZATION PLOTS] 그래픽 렌더링 컴포넌트 (st.pyplot 연동 리턴형 변환)
# ==============================================================================
def plot_observed_frame(wave_obs, flux_obs, sigma_obs, wave_range, step=10):
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
    return fig

def plot_rest_frame_original(wave_rest, flux_obs, sigma_obs, wave_range, step=10):
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
    return fig

def plot_dust_correction_comparison(wave_rest, flux_obs, flux_corr, wave_range):
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
    return fig

def plot_emission_lines_zoom(wave_rest, flux_corr, wave_range):
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
    return fig

# ==============================================================================
# [MENU 1] 사이드바 네비게이션 및 마스터 제어판 화면 구성
# ==============================================================================
st.sidebar.title("🌌 AstroFit 시스템")
menu = st.sidebar.radio(
    "이동할 페이지를 선택하세요:",
    ["1. 마스터 제어판 (Control Panel)", "2. pPXF 연속광 공제 설명", "3. Hβ 성분 분해 설명", "4. 비리얼 블랙홀 질량 계산"]
)

if menu == "1. 마스터 제어판 (Control Panel)":
    st.subheader("⚙️ Spectrum Analysis Report Master Control Panel")
    
    # 외부 링크 툴바 (HTML 스타일)
    st.markdown("""
    <div style="margin: 5px 0px 20px 0px; padding: 12px; background-color: #f8f9fa; border-left: 4px solid #1F4E79; border-radius: 4px;">
        <strong style="color: #1F4E79; font-size: 13px; display: block; margin-bottom: 8px; font-family:sans-serif;">
            외부 데이터베이스 및 템플릿 다운로드 빠른 링크 (New Tab)
        </strong>
        <a href="https://cas.sdss.org/dr19" target="_blank" style="text-decoration:none; background-color:#2E6B9E; color:white; padding:8px 14px; border-radius:4px; font-weight:bold; font-size:12px; margin-right:8px; display:inline-block;">🌌 SDSS DR19 CAS 바로가기</a>
        <a href="https://irsa.ipac.caltech.edu/applications/DUST/" target="_blank" style="text-decoration:none; background-color:#D97706; color:white; padding:8px 14px; border-radius:4px; font-weight:bold; font-size:12px; margin-right:8px; display:inline-block;">☁️ NASA IRSA Dust 조회</a>
        <a href="https://cloud.iac.es/index.php/s/aYECNyEQfqgYwt4?dir=/E-MILES" target="_blank" style="text-decoration:none; background-color:#059669; color:white; padding:8px 14px; border-radius:4px; font-weight:bold; font-size:12px; display:inline-block;">📚 E-MILES 템플릿 다운로드</a>
    </div>
    """, unsafe_allow_html=True)

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

    # 1단계: 제어판 데이터 세션 동기화 버튼
    if st.button("🔄 제어판 데이터 시스템 동기화 (Apply)", use_container_width=True):
        st.session_state.metadata["obj_name"] = obj_name.strip()
        st.session_state.metadata["ra"] = ra
        st.session_state.metadata["dec"] = dec
        st.session_state.metadata["obj_type"] = obj_type
        st.session_state.config["fits_file"] = fits_file
        st.session_state.config["emiles_file"] = emiles_file
        st.session_state.config["manual_Av"] = manual_Av
        st.session_state.config["Rv"] = Rv
        st.success("제어판 데이터 파라미터가 시스템 세션에 저장되었습니다!")

    # 데이터 상태 모니터 박스
    st.markdown("#### 🖥️ 현재 동기화된 시스템 데이터 상태")
    fits_status = st.session_state.config["fits_file"].name if st.session_state.config["fits_file"] else "❌ 미업로드 (No File)"
    emiles_status = st.session_state.config["emiles_file"].name if st.session_state.config["emiles_file"] else "❌ 미업로드 (No File)"
    summary_text = f"""객체: {st.session_state.metadata['obj_name']} | 좌표: ({st.session_state.metadata['ra']}, {st.session_state.metadata['dec']}) | 유형: {st.session_state.metadata['obj_type']}
[적용된 FITS 데이터]: {fits_status} | [적용된 템플릿]: {emiles_status}"""
    st.code(summary_text, language="text")

    # ==============================================================================
    # 💥 [PIPELINE EXECUTION] 1번째 분석 파이프라인 가동 제어부
    # ==============================================================================
    st.write("---")
    st.markdown("### 🚀 과학 연산 파이프라인 가동")
    
    if st.button("▶️ 적색편이 및 데이터 보정 파이프라인 가동", type="primary", use_container_width=True):
        if st.session_state.config["fits_file"] is None or st.session_state.config["emiles_file"] is None:
            st.error("❌ 에러: 제어판 [B] 구역에 FITS 파일과 E-MILES 템플릿 파일을 모두 업로드한 뒤 [동기화(Apply)]를 먼저 눌러주세요.")
        else:
            with st.spinner("⏳ 1단계 전처리 및 성간 소광 자동 연산 수행 중..."):
                try:
                    # 1. 템플릿 처리 (업로드된 바이트 객체 전달)
                    templates_list = setup_templates(st.session_state.config["emiles_file"])
                    st.write(f"✓ 검색 및 정렬된 FITS 템플릿 수: {len(templates_list)}개 로드 완료.")
                    
                    # 2. 전처리 연산 가동
                    w_obs, f_obs, s_obs, w_rest, f_corr, s_corr, calc_z, final_av = load_and_process_spectrum(
                        uploaded_fits=st.session_state.config["fits_file"],
                        manual_Av_str=st.session_state.config["manual_Av"],
                        Rv=st.session_state.config["Rv"]
                    )
                    
                    # 3. 전역 스트림 세션 상태 저장
                    st.session_state.pipeline_data_stream["templates"] = templates_list
                    st.session_state.pipeline_data_stream["wave_obs"] = w_obs
                    st.session_state.pipeline_data_stream["flux_obs"] = f_obs
                    st.session_state.pipeline_data_stream["sigma_obs"] = s_obs
                    st.session_state.pipeline_data_stream["wave_rest"] = w_rest
                    st.session_state.pipeline_data_stream["flux_dereddened"] = f_corr
                    st.session_state.pipeline_data_stream["sigma_dereddened"] = s_corr
                    st.session_state.pipeline_data_stream["z_calculated"] = calc_z
                    st.session_state.pipeline_data_stream["final_Av"] = final_av
                    st.session_state.pipeline_data_stream["is_ready"] = True
                    
                    st.success(f"🎉 1단계 파이프라인 가동 성공! 고유 적색편이값(z): {calc_z:.6f} | 최종 소광량(Av): {final_av:.4f} mag")
                    
                    # 4. 실시간 웹/앱 대시보드 시각화 출력 (Matplotlib 차트 4개 연속 렌더링)
                    st.markdown("#### 📉 실시간 물리 데이터 검수 그래프")
                    
                    st.pyplot(plot_observed_frame(w_obs, f_obs, s_obs, wave_range=OBS_WAVE_RANGE))
                    st.pyplot(plot_rest_frame_original(w_rest, f_obs, s_corr, wave_range=REST_WAVE_RANGE))
                    st.pyplot(plot_dust_correction_comparison(w_rest, f_obs, f_corr, wave_range=REST_WAVE_RANGE))
                    st.pyplot(plot_emission_lines_zoom(w_rest, f_corr, wave_range=ZOOM_WAVE_RANGE))
                    
                except Exception as e:
                    st.error(f"파이프라인 연산 중 치명적 오류 발생: {e}")

# ==============================================================================
# 설명 페이지 메뉴 목록 구조 보존
# ==============================================================================
elif menu == "2. pPXF 연속광 공제 설명":
    st.header("✨ pPXF 항성 연속광 공제")
    st.write("---")
    if st.session_state.pipeline_data_stream["is_ready"]:
        st.success(f"현재 로드된 천체 {st.session_state.metadata['obj_name']}의 전처리 데이터가 준비되어 있습니다.")
    else:
        st.info("1번 제어판에서 파이프라인을 먼저 가동해 주세요.")

elif menu == "3. Hβ 성분 분해 설명":
    st.header("📊 광폭 Hβ 방출선 성분 분해")
    st.write("---")

elif menu == "4. 비리얼 블랙홀 질량 계산":
    st.header("🕳️ 비리얼 정리 기반 블랙홀 질량 계산")
    st.write("---")
