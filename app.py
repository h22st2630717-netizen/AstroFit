import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# 1. 앱 페이지 기본 설정 (모바일 최적화 및 타이틀)
st.set_page_config(page_title="AstroFit", page_icon="🌌", layout="centered")

# 2. 왼쪽 사이드바 메뉴 구성 (호텔 앱의 왼쪽 메뉴 역할을 합니다)
st.sidebar.title("🌌 AstroFit Menu")
menu = st.sidebar.radio(
    "이동할 페이지를 선택하세요:",
    ["1. 스펙트럼 분석 실행", "2. pPXF 연속광 공제 설명", "3. Hβ 성분 분해 설명", "4. 비리얼 블랙홀 질량 계산"]
)

# 3. [메뉴 1] 실제 분석을 실행하는 페이지
if menu == "1. 스펙트럼 분석 실행":
    st.header("📈 스펙트럼 분석 실행")
    st.write("스펙트럼 데이터 파일(.fits 또는 .txt)을 업로드하여 분석을 시작하세요.")
    
    uploaded_file = st.file_uploader("파일을 선택해 주세요", type=["fits", "txt"])
    
    if uploaded_file is not None:
        st.success("파일 업로드 성공! 분석을 진행합니다.")
        # [여기에 나중에 사용자님의 진짜 코랩 스펙트럼 분석 코드가 들어갈 자리입니다]
        
        # 예시 그래프 출력
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x), color="purple")
        ax.set_title("Sample Spectrum Fit")
        st.pyplot(fig)

# 4. [메뉴 2] pPXF 설명 페이지
elif menu == "2. pPXF 연속광 공제 설명":
    st.header("✨ pPXF 항성 연속광 공제")
    st.write("---")
    st.subheader("💡 분석법 요약")
    st.write("Penalized Pixel-Fitting (pPXF) 알고리즘을 사용하여 은하의 항성 기원의 연속광(Stellar Continuum)을 정밀하게 모델링하고 이를 원본 스펙트럼에서 차감합니다.")
    st.info("이 과정을 거쳐야 순수한 활동성 은하핵(AGN) 기원의 방출선 프로파일을 얻을 수 있습니다.")

# 5. [메뉴 3] Hβ 분해 설명 페이지 (아까 구상했던 레이아웃 적용!)
elif menu == "3. Hβ 성분 분해 설명":
    st.header("📊 광폭 Hβ 방출선 성분 분해")
    st.write("---")
    st.subheader("💡 분석법 요약")
    st.write("차감 완료된 순수 방출선 데이터로부터 가우시안 성분 분해를 통해 좁은 성분(Narrow)과 넓은 성분(Broad)을 물리적으로 분리합니다.")
    st.write("**핵심 도출 물리량:** Broad $H\beta$ 성분의 FWHM(속도 폭) 및 광도(Luminosity)")
    # 나중에 여기에 실제 결과 샘플 이미지를 st.image()로 넣을 수 있습니다.

# 6. [메뉴 4] 비리얼 정리 설명 페이지
elif menu == "4. 비리얼 블랙홀 질량 계산":
    st.header("🕳️ 비리얼 정리 기반 블랙홀 질량 계산")
    st.write("---")
    st.subheader("💡 분석법 요약")
    st.write("도출된 $H\beta$ Broad 성분의 운동학 변수들을 Scaling 관계식에 대입하여 단일 에포크(Single-Epoch) 거대질량 블랙홀의 질량($M_{\bullet}$)을 최종 산출합니다.")
