import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder

# -------------------------------------------------------------------------
# 1. AI 앙상블 엔진 클래스 (기존 로직 유지, 경로 및 로드 방식 유연화)
# -------------------------------------------------------------------------
class CT_Ensemble_Engine:
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.m1 = Ridge(alpha=1.0)
        self.m2 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.m3 = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
        self.is_ready = False
        self.cat_vars = ['MA', 'SZ', 'IN', 'TH', 'DP']

    def train(self, uploaded_file):
        if uploaded_file is None:
            return "파일이 업로드되지 않았습니다."
        
        try:
            # 엑셀 로드 (헤더 2행 기준)
            df = pd.read_excel(uploaded_file, sheet_name='Past Data', header=1)
            df.columns = [str(c).strip().upper() for c in df.columns]
            
            # 컬럼명 설정
            target_col = 'POINCT'   # 과거 실측 CT
            past_nom_col = 'POMFCT' # 과거 해석 CT
            feature_cols = self.cat_vars + [past_nom_col]
            
            # 데이터 정제
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]
            
            # 범주형 데이터 인코딩
            X_enc = X.copy()
            X_enc[self.cat_vars] = self.encoder.fit_transform(X[self.cat_vars].astype(str))
            
            # 모델 학습
            self.m1.fit(X_enc, y)
            self.m2.fit(X_enc, y)
            self.m3.fit(X_enc, y)
            self.is_ready = True
            return "SUCCESS"
        except Exception as e:
            return f"학습 오류: {str(e)}"

    def predict(self, inputs):
        if not self.is_ready: 
            return None
        
        df_in = pd.DataFrame([{
            'MA': inputs['MA'], 'SZ': inputs['SZ'], 'IN': inputs['IN'],
            'TH': inputs['TH'], 'DP': inputs['DP'], 'POMFCT': inputs['NOMFCT']
        }])
        df_in[self.cat_vars] = self.encoder.transform(df_in[self.cat_vars].astype(str))
        
        pred = (self.m1.predict(df_in)[0] + self.m2.predict(df_in)[0] + self.m3.predict(df_in)[0]) / 3
        return pred

# -------------------------------------------------------------------------
# 2. 스트림릿 UI 메인 구성
# -------------------------------------------------------------------------
st.set_page_config(page_title="AI 사출 CT 정밀 분석 시스템", layout="centered")

st.title("AI 사출 정밀 예상 CT 시스템")
st.caption("과거 데이터를 기반으로 머신러닝 모델이 성형 해석 오차를 정밀 보정합니다.")
st.markdown("---")

# 세션 상태 초기화 (엔진 유지용)
if "engine" not in st.session_state:
    st.session_state.engine = CT_Ensemble_Engine()

# 파일 업로더 (웹 환경 맞춤 변경)
st.sidebar.header("📁 데이터 설정")
uploaded_file = st.sidebar.file_uploader("엑셀 파일(CT-INPUT-V6.xlsx)을 업로드하세요.", type=["xlsx"])

# 기본 MA 목록 선언
ma_list = ["650", "630", "8636"]

# 파일이 업로드되면 인공지능 학습 실행
if uploaded_file is not None:
    with st.spinner("AI 엔진 학습 진행 중..."):
        status = st.session_state.engine.train(uploaded_file)
    
    if status == "SUCCESS":
        st.sidebar.success("✅ AI 엔진 학습 성공!")
        # 업로드된 파일에서 MA 목록 추출해오기
        try:
            temp_df = pd.read_excel(uploaded_file, sheet_name='Past Data', header=1)
            ma_list = sorted([str(x).strip() for x in temp_df['MA'].dropna().unique()])
        except:
            pass
    else:
        st.sidebar.error(f"❌ 학습 실패: {status}")
else:
    st.sidebar.warning("⚠️ 작동을 위해 엑셀 파일을 먼저 업로드해주세요.")

# STEP 1. 현재 공정 및 해석 조건 입력 영역
st.header("STEP 1. 현재 공정 및 해석 조건 입력")

col1, col2 = st.columns(2)

with col1:
    ma_val = st.selectbox("MA", options=ma_list)
    sz_val = st.selectbox("SZ", options=["S", "M", "L"])
    in_val = st.selectbox("IN", options=["IO", "IX"])

with col2:
    th_val = st.selectbox("TH", options=["TS", "TM", "TL"])
    dp_val = st.selectbox("DP", options=["DS", "DM", "DL"])
    
    # 숫자 입력창 (기존 tk.Entry 대체)
    nomfct_val = st.number_input("현재 성형 해석 후 CT (NOMFCT)", min_value=0.0, value=200.0, step=0.1)

st.markdown("---")

# STEP 2. 결과 출력 영역
st.header("STEP 2. 현재 예상 사출 전체 CT (NOPRECT)")

# 분석 실행 버튼
if st.button("AI 정밀 분석 실행 (NOPRECT)", type="primary", use_container_width=True):
    if not st.session_state.engine.is_ready:
        st.error("먼저 왼쪽 사이드바에서 엑셀 파일을 업로드하여 AI 엔진을 학습시켜 주세요.")
    else:
        try:
            input_data = {
                "MA": ma_val, "SZ": sz_val, "IN": in_val,
                "TH": th_val, "DP": dp_val, "NOMFCT": nomfct_val
            }
            
            # 예측 값 계산
            result = st.session_state.engine.predict(input_data)
            
            if result is not None:
                # 결과 카드 디자인 시각화
                st.success(f"### 🎉 최종 예상 CT (NOPRECT): **{result:.2f} s**")
                st.info(f"💡 현재 입력하신 성형 해석치 **{nomfct_val}s** 대비 AI 오차 보정이 완료되었습니다.")
            else:
                st.error("예측에 실패했습니다. 데이터를 다시 확인해주세요.")
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
