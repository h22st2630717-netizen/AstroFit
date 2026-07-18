import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits  # FITS 파일 처리를 위해 추가

# 1. 앱 페이지 설정 (상단 타이틀 및 모바일 레이아웃)
st.set_page_config(page_title="AstroFit", page_icon="🌌", layout="centered")

st.sidebar.title("🌌 AstroFit 시스템")
menu = st.sidebar.radio(
    "메뉴를 선택하세요:",
    ["1. 스펙트럼 분석 실행", "2. pPXF 연속광 공제 설명", "3. Hβ 성분 분해 설명", "4. 비리얼 블랙홀 질량 계산"]
)

# =========================================================================
# [핵심] 코랩에서 작성하신 천문학 분석 함수들을 여기에 정의합니다.
# =========================================================================

def run_ppxf_subtraction(spectrum_data):
    """ 🔴 1. pPXF 항성 연속광 공제 알고리즘 자리 """
    # [코랩 코드 입력창]
    # 여기에 원래 쓰시던 pPXF 피팅 및 연속광 차감 코드를 넣으세요.
    # 예시로 받은 데이터를 그대로 리턴하는 뼈대만 잡아둡니다.
    pure_emission_flux = spectrum_data  
    return pure_emission_flux

def fit_hbeta_components(wave, flux):
    """ 🔴 2. H-beta 다중 가우시안 성분 분해 알고리즘 자리 """
    # [코랩 코드 입력창]
    # 여기에 비선형 최소제곱법(curve_fit 등)을 이용한 가우시안 피팅 코드를 넣으세요.
    # 결과값으로 대략적인 FWHM과 광도(Luminosity)가 계산되어 나온다고 가정합니다.
    
    sample_fwhm = 4850.0  # 예시 값 (실제 계산된 값으로 대체)
    sample_log_l = 42.32  # 예시 값 (실제 계산된 값으로 대체)
    return sample_fwhm, sample_log_l

def calculate_black_hole_mass(fwhm, log_l):
    """ 🔴 3. 비리얼 정리 기반 블랙홀 질량 계산 자리 """
    # [코랩 코드 입력창]
    # Greene & Ho (2005) 등의 Scaling 관계식을 여기에 코딩하세요.
    # log_M = a + b * log(FWHM) + c * log(L)
    log_bh_mass = 0.89 * log_l + 2.0 * np.log10(fwhm) - 4.1  # 예시 수식
    return log_bh_mass


# =========================================================================
# 각 메뉴별 화면 구현 및 실행 레이아웃
# =========================================================================

if menu == "1. 스펙트럼 분석 실행":
    st.header("📈 실시간 스펙트럼 분석 파이프라인")
    st.write("관측된 천체의 스펙트럼 데이터(.fits)를 업로드하면 전처리부터 질량 계산까지 자동으로 진행됩니다.")
    
    # 파일 업로더 생성
    uploaded_file = st.file_uploader("FITS 스펙트럼 파일 업로드", type=["fits", "fits.gz"])
    
    if uploaded_file is not None:
        with st.spinner("천문 데이터 분석 중... 잠시만 기다려주세요."):
            try:
                # [중요] 스트림릿에서 FITS 파일을 읽는 표준 방법입니다.
                with fits.open(uploaded_file) as hdul:
                    # 데이터 구조에 맞게 인덱스(0 또는 1)를 조정하세요.
                    data = hdul[0].data  
                    header = hdul[0].header
                
                st.success("데이터 로드 완료! 파이프라인을 가동합니다.")
                
                # ---- 파이프라인 실시간 가동 ----
                # 1단계: pPXF 연속광 차감
                pure_flux = run_ppxf_subtraction(data)
                
                # 2단계: 가우시안 성분 분해 (FWHM, 광도 도출)
                # 임의의 파장축 x를 생성하거나 FITS 헤더에서 계산해와야 합니다.
                wave = np.linspace(4000, 5000, len(data)) 
                fwhm, log_l = fit_hbeta_components(wave, pure_flux)
                
                # 3단계: 블랙홀 질량 계산
                log_mbh = calculate_black_hole_mass(fwhm, log_l)
                
                # ---- 결과 대시보드 시각화 ----
                st.write("---")
                st.subheader("📊 핵심 분석 결과 (Physical Parameters)")
                
                # 모바일 화면에서 보기 좋게 3열로 수치 표시
                col1, col2, col3 = st.columns(3)
                col1.metric("Hβ FWHM", f"{fwhm:.1f} km/s")
                col2.metric("Log L (Hβ)", f"{log_l:.2f} erg/s")
                col3.metric("Black Hole Mass", f"10^{log_mbh:.2f} M☉")
                
                # ---- 최종 피팅 그래프 출력 ----
                st.write("---")
                st.subheader("📉 Spectral Decomposition Plot")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(wave, data, label="Observed", color="gray", alpha=0.5)
                ax.plot(wave, pure_flux, label="Pure Emission Line", color="purple")
                ax.set_xlabel("Rest-frame Wavelength (Å)")
                ax.set_ylabel("Flux Intensity")
                ax.legend()
                
                # 스트림릿 전용 그래프 출력 함수 (핵심)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"분석 중 에러가 발생했습니다: {e}")
                st.info("FITS 파일의 데이터 구조(HDU index)나 데이터 포맷을 확인해 주세요.")

# --- 설명 페이지 메뉴들 (기존 구조 유지) ---
elif menu == "2. pPXF 연속광 공제 설명":
    st.header("✨ pPXF 항성 연속광 공제")
    st.write("---")
    st.subheader("💡 분석법 요약")
    st.write("Penalized Pixel-Fitting (pPXF) 알고리즘을 사용하여 은하의 항성 기원의 연속광(Stellar Continuum)을 정밀하게 모델링하고 이를 원본 스펙트럼에서 차감합니다.")

elif menu == "3. Hβ 성분 분해 설명":
    st.header("📊 광폭 Hβ 방출선 성분 분해")
    st.write("---")
    st.subheader("💡 분석법 요약")
    st.write("차감 완료된 순수 방출선 데이터로부터 가우시안 성분 분해를 통해 좁은 성분(Narrow)과 넓은 성분(Broad)을 물리적으로 분리합니다.")

elif menu == "4. 비리얼 블랙홀 질량 계산":
    st.header("🕳️ 비리얼 정리 기반 블랙홀 질량 계산")
    st.write("---")
    st.subheader("💡 분석법 요약")
    st.write("도출된 Hβ Broad 성분의 운동학 변수들을 Scaling 관계식에 대입하여 단일 에포크(Single-Epoch) 거대질량 블랙홀의 질량을 최종 산출합니다.")
