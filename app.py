# ============================================
# app.py - STREAMLIT WEB ARAYÜZÜ (DÜZELTİLMİŞ)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="🚗 Araba Fiyat Tahmini",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# VERİ YÜKLEME
# ============================================

@st.cache_data
def load_data():
    """Veri setini yükle ve temizle"""
    df = pd.read_csv('neuauto.csv')
    
    # Fiyat temizleme
    def temizle_fiyat(fiyat):
        if pd.isna(fiyat):
            return np.nan
        fiyat_str = str(fiyat)
        fiyat_str = fiyat_str.replace(' TL', '').replace('.', '').replace(',', '.')
        try:
            return float(fiyat_str)
        except:
            return np.nan
    
    # Kilometre temizleme
    def temizle_kilometre(km):
        if pd.isna(km):
            return np.nan
        km_str = str(km)
        km_str = km_str.replace(' km', '').replace('.', '').replace(',', '')
        try:
            return float(km_str)
        except:
            return np.nan
    
    df['fiyat_temiz'] = df['fiyat'].apply(temizle_fiyat)
    df['kilometre_temiz'] = df['kilometre'].apply(temizle_kilometre)
    df['yil'] = pd.to_numeric(df['yil'], errors='coerce')
    df['boyali_sayisi'] = pd.to_numeric(df['boyali_sayisi'], errors='coerce').fillna(0)
    df['degisen_sayisi'] = pd.to_numeric(df['degisen_sayisi'], errors='coerce').fillna(0)
    
    # Kategorik sütunları string yap ve NaN temizle
    for col in ['marka', 'seri', 'model', 'vites_tipi', 'kimden', 'konum']:
        if col in df.columns:
            df[col] = df[col].fillna('Bilinmiyor').astype(str)
    
    # Aykırı değerleri ve NaN'ları temizle
    df = df.dropna(subset=['fiyat_temiz', 'kilometre_temiz', 'yil'])
    
    df = df[
        (df['fiyat_temiz'] >= 50000) & 
        (df['fiyat_temiz'] <= 15000000) &
        (df['kilometre_temiz'] >= 0) &
        (df['kilometre_temiz'] <= 1000000) &
        (df['yil'] >= 1990) &
        (df['yil'] <= 2025)
    ].copy()
    
    # 50'den az olan serileri filtrele
    seri_sayilari = df['seri'].value_counts()
    gecerli_seriler = seri_sayilari[seri_sayilari >= 50].index
    df = df[df['seri'].isin(gecerli_seriler)]
    
    return df

@st.cache_resource
def load_model():
    """Model yükle"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        marka_seri_mapping = joblib.load('models/marka_seri_mapping.pkl')
        model_info = joblib.load('models/model_info.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        return model, scaler, label_encoders, marka_seri_mapping, model_info, feature_columns, True
    except Exception as e:
        return None, None, None, {}, {}, [], False

# ============================================
# TAHMİN FONKSİYONU
# ============================================

def predict_price(model, scaler, label_encoders, marka, seri, yil, kilometre, 
                  vites_tipi, kimden, boyali_sayisi, degisen_sayisi):
    
    arac_yasi = 2025 - yil
    km_yas_orani = kilometre / (arac_yasi + 1)
    toplam_hasar = boyali_sayisi + degisen_sayisi
    hasarsiz = 1 if (boyali_sayisi == 0 and degisen_sayisi == 0) else 0
    
    try:
        marka_encoded = label_encoders['marka'].transform([marka])[0]
    except:
        marka_encoded = 0
    try:
        seri_encoded = label_encoders['seri'].transform([seri])[0]
    except:
        seri_encoded = 0
    try:
        vites_encoded = label_encoders['vites_tipi'].transform([vites_tipi])[0]
    except:
        vites_encoded = 0
    try:
        kimden_encoded = label_encoders['kimden'].transform([kimden])[0]
    except:
        kimden_encoded = 0
    
    features = np.array([[
        kilometre, yil, boyali_sayisi, degisen_sayisi,
        arac_yasi, km_yas_orani, toplam_hasar, hasarsiz,
        marka_encoded, seri_encoded, vites_encoded, kimden_encoded
    ]])
    
    features_scaled = scaler.transform(features)
    prediction_log = model.predict(features_scaled)[0]
    prediction = np.expm1(prediction_log)
    
    return prediction

# ============================================
# ANA UYGULAMA
# ============================================

def main():
    st.markdown('<h1 class="main-header">🚗 Araba Fiyat Tahmini</h1>', unsafe_allow_html=True)
    
    # Veri yükle
    try:
        df = load_data()
    except Exception as e:
        st.error(f"❌ Veri yüklenemedi: {e}")
        st.stop()
    
    # Model yükle
    model, scaler, label_encoders, marka_seri_mapping, model_info, feature_columns, model_loaded = load_model()
    
    # Sidebar
    st.sidebar.title("🔧 Menü")
    page = st.sidebar.radio(
        "Sayfa Seçin:",
        ["🏠 Ana Sayfa", "🔮 Fiyat Tahmini", "📊 Veri Analizi", "📈 Model Performansı"]
    )
    
    # ============================================
    # ANA SAYFA
    # ============================================
    if page == "🏠 Ana Sayfa":
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Toplam İlan", f"{len(df):,}")
        col2.metric("🚗 Marka Sayısı", f"{df['marka'].nunique()}")
        col3.metric("💰 Ortalama Fiyat", f"{df['fiyat_temiz'].mean():,.0f} TL")
        col4.metric("📅 Yıl Aralığı", f"{int(df['yil'].min())}-{int(df['yil'].max())}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚗 En Çok İlan Verilen Markalar")
            marka_counts = df['marka'].value_counts().head(10)
            fig = px.bar(
                x=marka_counts.values, y=marka_counts.index, orientation='h',
                color=marka_counts.values, color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_title="İlan Sayısı", yaxis_title="", showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💰 En Pahalı Markalar")
            marka_fiyat = df.groupby('marka')['fiyat_temiz'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=marka_fiyat.values, y=marka_fiyat.index, orientation='h',
                color=marka_fiyat.values, color_continuous_scale='RdYlGn'
            )
            fig.update_layout(xaxis_title="Ortalama Fiyat (TL)", yaxis_title="", showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Yıllara Göre İlan Sayısı")
            yil_counts = df['yil'].value_counts().sort_index()
            fig = px.line(x=yil_counts.index, y=yil_counts.values, markers=True)
            fig.update_layout(xaxis_title="Yıl", yaxis_title="İlan Sayısı", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚙️ Vites Tipi Dağılımı")
            vites_counts = df['vites_tipi'].value_counts()
            fig = px.pie(values=vites_counts.values, names=vites_counts.index, hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # FİYAT TAHMİNİ
    # ============================================
    elif page == "🔮 Fiyat Tahmini":
        st.markdown("---")
        
        if not model_loaded:
            st.warning("⚠️ Model yüklenemedi. Sadece veri analizi yapılabilir.")
            st.stop()
        
        st.subheader("🔮 Araç Bilgilerini Girin")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            markalar = sorted(df['marka'].unique().tolist())
            marka = st.selectbox("🚗 Marka", markalar)
            
            if marka in marka_seri_mapping:
                seriler = [str(s) for s in marka_seri_mapping[marka] if pd.notna(s)]
                seriler = sorted(set(seriler))
            else:
                seriler = sorted(df[df['marka'] == marka]['seri'].unique().tolist())
            
            if not seriler:
                seriler = ["Bilinmiyor"]
            seri = st.selectbox("📋 Seri", seriler)
            
            yil = st.slider("📅 Model Yılı", 
                           min_value=int(df['yil'].min()), 
                           max_value=int(df['yil'].max()), 
                           value=2020)
        
        with col2:
            kilometre = st.number_input("🛣️ Kilometre", min_value=0, max_value=500000, value=50000, step=5000)
            
            vites_tipleri = sorted(df['vites_tipi'].unique().tolist())
            vites_tipi = st.selectbox("⚙️ Vites Tipi", vites_tipleri)
            
            kimden_options = sorted(df['kimden'].unique().tolist())
            kimden = st.selectbox("👤 Kimden", kimden_options)
        
        with col3:
            boyali_sayisi = st.slider("🎨 Boyalı Parça", min_value=0, max_value=13, value=0)
            degisen_sayisi = st.slider("🔧 Değişen Parça", min_value=0, max_value=13, value=0)
        
        st.markdown("---")
        
        if st.button("🔮 Fiyat Tahmin Et", type="primary", use_container_width=True):
            try:
                tahmin = predict_price(
                    model, scaler, label_encoders, marka, seri, yil,
                    kilometre, vites_tipi, kimden, boyali_sayisi, degisen_sayisi
                )
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>💰 Tahmini Fiyat</h2>
                    <h1 style="font-size: 3rem;">{tahmin:,.0f} TL</h1>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("📋 Araç Özellikleri")
                col1, col2 = st.columns(2)
                col1.write(f"**Marka:** {marka} | **Seri:** {seri} | **Yıl:** {yil}")
                col2.write(f"**KM:** {kilometre:,} | **Vites:** {vites_tipi} | **Hasar:** {boyali_sayisi+degisen_sayisi} parça")
                
            except Exception as e:
                st.error(f"❌ Hata: {e}")
    
    # ============================================
    # VERİ ANALİZİ
    # ============================================
    elif page == "📊 Veri Analizi":
        st.markdown("---")
        st.subheader("🔍 Filtreler")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            markalar_list = sorted(df['marka'].unique().tolist())
            selected_marka = st.multiselect("Marka", options=markalar_list, default=[])
        
        with col2:
            yil_min = int(df['yil'].min())
            yil_max = int(df['yil'].max())
            yil_range = st.slider("Yıl Aralığı", min_value=yil_min, max_value=yil_max, value=(yil_min, yil_max))
        
        with col3:
            vites_list = sorted(df['vites_tipi'].unique().tolist())
            selected_vites = st.multiselect("Vites Tipi", options=vites_list, default=[])
        
        # Filtreleme
        df_filtered = df.copy()
        if selected_marka:
            df_filtered = df_filtered[df_filtered['marka'].isin(selected_marka)]
        df_filtered = df_filtered[df_filtered['yil'].between(yil_range[0], yil_range[1])]
        if selected_vites:
            df_filtered = df_filtered[df_filtered['vites_tipi'].isin(selected_vites)]
        
        st.info(f"📊 Filtrelenmiş kayıt: {len(df_filtered):,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Fiyat Dağılımı")
            fig = px.histogram(df_filtered, x='fiyat_temiz', nbins=50)
            fig.update_layout(xaxis_title="Fiyat (TL)", yaxis_title="Frekans")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🛣️ Kilometre Dağılımı")
            fig = px.histogram(df_filtered, x='kilometre_temiz', nbins=50, color_discrete_sequence=['#e74c3c'])
            fig.update_layout(xaxis_title="Kilometre", yaxis_title="Frekans")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Veri Önizleme")
        st.dataframe(df_filtered[['marka', 'seri', 'yil', 'kilometre_temiz', 'vites_tipi', 'fiyat_temiz']].head(50))
    
    # ============================================
    # MODEL PERFORMANSI
    # ============================================
    elif page == "📈 Model Performansı":
        st.markdown("---")
        
        if not model_loaded:
            st.warning("⚠️ Model yüklenemedi.")
            st.stop()
        
        st.subheader("📊 Model Metrikleri")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{model_info.get('test_r2', 0):.4f}")
        col2.metric("RMSE", f"{model_info.get('test_rmse', 0):,.0f} TL")
        col3.metric("MAE", f"{model_info.get('test_mae', 0):,.0f} TL")
        col4.metric("MAPE", f"{model_info.get('test_mape', 0):.2f}%")
        
        st.markdown("---")
        
        if hasattr(model, 'feature_importances_') and feature_columns:
            st.subheader("📊 Özellik Önemliliği")
            importance_df = pd.DataFrame({
                'Özellik': feature_columns,
                'Önem': model.feature_importances_
            }).sort_values('Önem', ascending=True)
            
            fig = px.bar(importance_df, x='Önem', y='Özellik', orientation='h',
                        color='Önem', color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
