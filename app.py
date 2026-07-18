import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# [페이지 초기 설정 및 세션 상태(메모리) 초기화]
# ==============================================================================
st.set_page_config(page_title="AstroFit", page_icon="🌌", layout="wide")

# 파일 객체 자체를 메모리에 저장하기 위해 세션 상태를 수정합니다.
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

# ==============================================================================
# [사이드바 메뉴] 네비게이션
# ==============================================================================
st.sidebar.title("🌌 AstroFit 시스템")
menu = st.sidebar.radio(
    "이동할 페이지를 선택하세요:",
    ["1. 마스터 제어판 (Control Panel)", "2. pPXF 연속광 공제 설명", "3. Hβ 성분 분해 설명", "4. 비리얼 블랙홀 질량 계산"]
)

# ==============================================================================
# [메뉴 1] 마스터 제어판 화면 구현
# ==============================================================================
if menu == "1. 마스터 제어판 (Control Panel)":
    st.subheader("⚙️ Spectrum Analysis Report Master Control Panel")
    
    # [UI COMPONENTS - EXTERNAL LINKS TOOLBAR] 외부 데이터베이스 바로가기 링크 툴바
    st.markdown("""
    <div style="margin: 5px 0px 20px 0px; padding: 12px; background-color: #f8f9fa; border-left: 4px solid #1F4E79; border-radius: 4px;">
        <strong style="color: #1F4E79; font-size: 13px; display: block; margin-bottom: 8px; font-family:sans-serif;">
            외부 데이터베이스 및 템플릿 다운로드 빠른 링크 (New Tab)
        </strong>
        <a href="https://cas.sdss.org/dr19" target="_blank"
           style="text-decoration:none; background-color:#2E6B9E; color:white; padding:8px 14px; border-radius:4px; font-weight:bold; font-size:12px; margin-right:8px; display:inline-block;">
            🌌 SDSS DR19 CAS 바로가기
        </a>
        <a href="https://irsa.ipac.caltech.edu/applications/DUST/" target="_blank"
           style="text-decoration:none; background-color:#D97706; color:white; padding:8px 14px; border-radius:4px; font-weight:bold; font-size:12px; margin-right:8px; display:inline-block;">
            ☁️ NASA IRSA Dust 조회
        </a>
        <a href="https://cloud.iac.es/index.php/s/aYECNyEQfqgYwt4?dir=/E-MILES" target="_blank"
           style="text-decoration:none; background-color:#059669; color:white; padding:8px 14px; border-radius:4px; font-weight:bold; font-size:12px; display:inline-block;">
            📚 E-MILES 템플릿 다운로드
        </a>
    </div>
    """, unsafe_allow_html=True)

    # 대시보드 폼 요소 생성 (좌우 분할)
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
        
        # 💡 핵심 변경 항목: 경로 입력창을 파일 업로더(st.file_uploader)로 전면 교체!
        fits_file = st.file_uploader("SDSS FITS 파일 선택 (.fits)", type=["fits", "fits.gz"])
        emiles_file = st.file_uploader("E-MILES 템플릿 파일 선택 (.tar.gz)", type=["gz", "tar.gz"])
        
        manual_Av = st.text_input("성간소광량 Av (or None):", value=st.session_state.config["manual_Av"])
        Rv = st.number_input("Rv 상수 (기본 3.1):", value=st.session_state.config["Rv"], format="%.1f")

    # 설정 적용 버튼 (동기화 시스템)
    if st.button("🔄 제어판 데이터 시스템 동기화 (Apply)", use_container_width=True):
        st.session_state.metadata["obj_name"] = obj_name.strip()
        st.session_state.metadata["ra"] = ra
        st.session_state.metadata["dec"] = dec
        st.session_state.metadata["obj_type"] = obj_type

        # 💡 업로드된 파일 객체 자체를 세션에 바인딩합니다.
        st.session_state.config["fits_file"] = fits_file
        st.session_state.config["emiles_file"] = emiles_file
        st.session_state.config["manual_Av"] = manual_Av.strip()
        st.session_state.config["Rv"] = Rv

        if fits_file and emiles_file:
            st.success("FITS 및 템플릿 파일이 분석 메모리에 성공적으로 로드되었습니다!")
        else:
            st.warning("설정은 세이브되었으나, 파일이 아직 업로드되지 않았습니다. 파일을 선택해 주세요.")
        
    # [OUTPUT BOX] 현재 파일 업로드 상태 및 메타데이터 실시간 요약 출력
    st.markdown("#### 🖥️ 현재 동기화된 시스템 데이터 상태")
    
    # 파일 상태 문자열 처리
    fits_status = st.session_state.config["fits_file"].name if st.session_state.config["fits_file"] else "❌ 미업로드 (No File)"
    emiles_status = st.session_state.config["emiles_file"].name if st.session_state.config["emiles_file"] else "❌ 미업로드 (No File)"
    av_display = "None" if st.session_state.config["manual_Av"].upper() == "NONE" else f"{st.session_state.config['manual_Av']} mag"
    
    summary_text = f"""객체: {st.session_state.metadata['obj_name']} | 좌표: ({st.session_state.metadata['ra']}, {st.session_state.metadata['dec']}) | 유형: {st.session_state.metadata['obj_type']}
[적용된 FITS 데이터]: {fits_status}
[적용된 E-MILES 템플릿]: {emiles_status}
소광 설정치: Av = {av_display} (Rv = {st.session_state.config['Rv']})"""
    
    st.code(summary_text, language="text")

# ==============================================================================
# [나머지 메뉴들] 설명 페이지 프레임 유지
# ==============================================================================
elif menu == "2. pPXF 연속광 공제 설명":
    st.header("✨ pPXF 항성 연속광 공제")
    st.write("---")
    st.write(f"현재 선택된 천체: **{st.session_state.metadata['obj_name']}**")
    st.write("Penalized Pixel-Fitting (pPXF) 알고리즘 설명 및 연동 공간입니다.")

elif menu == "3. Hβ 성분 분해 설명":
    st.header("📊 광폭 Hβ 방출선 성분 분해")
    st.write("---")
    st.write("차감 완료된 순수 방출선 데이터 가우시안 성분 분해 상세 설명 공간입니다.")

elif menu == "4. 비리얼 블랙홀 질량 계산":
    st.header("🕳️ 비리얼 정리 기반 블랙홀 질량 계산")
    st.write("---")
    st.write("Scaling 관계식을 적용하여 단일 에포크 블랙홀 질량을 최종 산출하는 공간입니다.")
