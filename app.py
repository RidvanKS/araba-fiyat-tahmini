# ============================================
# app.py - STREAMLIT WEB ARAYÜZÜ
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# VERİ VE MODEL YÜKLEME
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
    df['boyali_sayisi'] = pd.to_numeric(df['boyali_sayisi'], errors='coerce')
    df['degisen_sayisi'] = pd.to_numeric(df['degisen_sayisi'], errors='coerce')
    
    # Aykırı değerleri temizle
    FIYAT_MIN = 50_000
    FIYAT_MAX = 15_000_000
    KM_MAX = 1_000_000
    
    df = df[
        (df['fiyat_temiz'] >= FIYAT_MIN) & 
        (df['fiyat_temiz'] <= FIYAT_MAX) &
        (df['kilometre_temiz'] <= KM_MAX)
    ]
    
    # 50'den az olan serileri filtrele
    seri_sayilari = df['seri'].value_counts()
    gecerli_seriler = seri_sayilari[seri_sayilari >= 50].index
    df = df[df['seri'].isin(gecerli_seriler)]
    
    return df

@st.cache_resource
def load_model():
    """Model ve gerekli objeleri yükle"""
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    marka_seri_mapping = joblib.load('models/marka_seri_mapping.pkl')
    model_info = joblib.load('models/model_info.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    
    return model, scaler, label_encoders, marka_seri_mapping, model_info, feature_columns

# ============================================
# TAHMİN FONKSİYONU
# ============================================

def predict_price(model, scaler, label_encoders, feature_columns,
                  marka, seri, yil, kilometre, vites_tipi, kimden, 
                  boyali_sayisi, degisen_sayisi):
    """Fiyat tahmini yap"""
    
    # Araç yaşı hesapla
    arac_yasi = 2025 - yil
    
    # Kilometre/Yaş oranı
    km_yas_orani = kilometre / (arac_yasi + 1)
    
    # Toplam hasar
    toplam_hasar = boyali_sayisi + degisen_sayisi
    
    # Hasarsız mı?
    hasarsiz = 1 if (boyali_sayisi == 0 and degisen_sayisi == 0) else 0
    
    # Label Encoding
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
    
    # Feature array oluştur
    features = np.array([[
        kilometre,           # kilometre_temiz
        yil,                 # yil
        boyali_sayisi,       # boyali_sayisi
        degisen_sayisi,      # degisen_sayisi
        arac_yasi,           # arac_yasi
        km_yas_orani,        # km_yas_orani
        toplam_hasar,        # toplam_hasar
        hasarsiz,            # hasarsiz
        marka_encoded,       # marka_encoded
        seri_encoded,        # seri_encoded
        vites_encoded,       # vites_tipi_encoded
        kimden_encoded       # kimden_encoded
    ]])
    
    # Ölçeklendirme
    features_scaled = scaler.transform(features)
    
    # Tahmin (log scale'den geri dönüşüm)
    prediction_log = model.predict(features_scaled)[0]
    prediction = np.expm1(prediction_log)
    
    return prediction

# ============================================
# ANA UYGULAMA
# ============================================

def main():
    # Başlık
    st.markdown('<h1 class="main-header">🚗 Araba Fiyat Tahmini Uygulaması</h1>', unsafe_allow_html=True)
    
    # Veri ve model yükle
    try:
        df = load_data()
        model, scaler, label_encoders, marka_seri_mapping, model_info, feature_columns = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"⚠️ Model veya veri yüklenirken hata oluştu: {str(e)}")
        model_loaded = False
        df = load_data()
    
    # Sidebar
    st.sidebar.title("🔧 Navigasyon")
    page = st.sidebar.radio(
        "Sayfa Seçin:",
        ["🏠 Ana Sayfa", "🔮 Fiyat Tahmini", "📊 Veri Analizi", "📈 Model Performansı", "ℹ️ Hakkında"]
    )
    
    # ============================================
    # ANA SAYFA
    # ============================================
    if page == "🏠 Ana Sayfa":
        st.markdown("---")
        
        # Özet Metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Toplam İlan",
                value=f"{len(df):,}",
                delta="Aktif"
            )
        
        with col2:
            st.metric(
                label="🚗 Marka Sayısı",
                value=f"{df['marka'].nunique()}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="💰 Ortalama Fiyat",
                value=f"{df['fiyat_temiz'].mean():,.0f} TL",
                delta=None
            )
        
        with col4:
            st.metric(
                label="📅 Yıl Aralığı",
                value=f"{int(df['yil'].min())}-{int(df['yil'].max())}",
                delta=None
            )
        
        st.markdown("---")
        
        # Grafik satırı
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚗 En Çok İlan Verilen Markalar")
            marka_counts = df['marka'].value_counts().head(10)
            fig = px.bar(
                x=marka_counts.values,
                y=marka_counts.index,
                orientation='h',
                color=marka_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_title="İlan Sayısı",
                yaxis_title="Marka",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💰 Marka Bazında Ortalama Fiyat")
            marka_fiyat = df.groupby('marka')['fiyat_temiz'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=marka_fiyat.values,
                y=marka_fiyat.index,
                orientation='h',
                color=marka_fiyat.values,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(
                xaxis_title="Ortalama Fiyat (TL)",
                yaxis_title="Marka",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # İkinci grafik satırı
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Yıllara Göre İlan Dağılımı")
            yil_counts = df['yil'].value_counts().sort_index()
            fig = px.line(
                x=yil_counts.index,
                y=yil_counts.values,
                markers=True
            )
            fig.update_layout(
                xaxis_title="Yıl",
                yaxis_title="İlan Sayısı",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚙️ Vites Tipi Dağılımı")
            vites_counts = df['vites_tipi'].value_counts()
            fig = px.pie(
                values=vites_counts.values,
                names=vites_counts.index,
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # FİYAT TAHMİNİ SAYFASI
    # ============================================
    elif page == "🔮 Fiyat Tahmini":
        st.markdown("---")
        st.subheader("🔮 Araç Bilgilerini Girin")
        
        if not model_loaded:
            st.error("⚠️ Model yüklenmedi! Lütfen önce modeli eğitin ve kaydedin.")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Marka seçimi
            markalar = sorted(df['marka'].dropna().astype(str).unique().tolist())
            marka = st.selectbox("🚗 Marka", markalar)
            
            # Seçilen markaya göre seri filtreleme
            if marka in marka_seri_mapping:
                seriler = sorted(marka_seri_mapping[marka])
            else:
                seriler = sorted(df[df['marka'] == marka]['seri'].unique())
            
            seri = st.selectbox("📋 Seri", seriler)
            
            # Yıl
            yil = st.slider(
                "📅 Model Yılı",
                min_value=int(df['yil'].min()),
                max_value=int(df['yil'].max()),
                value=2020
            )
        
        with col2:
            # Kilometre
            kilometre = st.number_input(
                "🛣️ Kilometre",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=5000
            )
            
            # Vites tipi
            vites_tipleri = sorted(df['vites_tipi'].dropna().astype(str).unique().tolist())
            vites_tipi = st.selectbox("⚙️ Vites Tipi", vites_tipleri)
            
            # Kimden
            kimden_options = sorted(df['kimden'].dropna().astype(str).unique().tolist())
            kimden = st.selectbox("👤 Kimden", kimden_options)
        
        with col3:
            # Boyalı sayısı
            boyali_sayisi = st.slider(
                "🎨 Boyalı Parça Sayısı",
                min_value=0,
                max_value=13,
                value=0
            )
            
            # Değişen sayısı
            degisen_sayisi = st.slider(
                "🔧 Değişen Parça Sayısı",
                min_value=0,
                max_value=13,
                value=0
            )
        
        st.markdown("---")
        
        # Tahmin butonu
        if st.button("🔮 Fiyat Tahmin Et", type="primary", use_container_width=True):
            with st.spinner("Tahmin yapılıyor..."):
                try:
                    tahmin = predict_price(
                        model, scaler, label_encoders, feature_columns,
                        marka, seri, yil, kilometre, vites_tipi, kimden,
                        boyali_sayisi, degisen_sayisi
                    )
                    
                    # Sonuç gösterimi
                    st.markdown("---")
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>💰 Tahmini Fiyat</h2>
                        <h1>{tahmin:,.0f} TL</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detaylar
                    st.markdown("---")
                    st.subheader("📋 Araç Özellikleri")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.info(f"**Marka:** {marka}")
                        st.info(f"**Seri:** {seri}")
                    
                    with col2:
                        st.info(f"**Yıl:** {yil}")
                        st.info(f"**Kilometre:** {kilometre:,} km")
                    
                    with col3:
                        st.info(f"**Vites:** {vites_tipi}")
                        st.info(f"**Kimden:** {kimden}")
                    
                    with col4:
                        st.info(f"**Boyalı:** {boyali_sayisi} parça")
                        st.info(f"**Değişen:** {degisen_sayisi} parça")
                    
                    # Benzer araçlar
                    st.markdown("---")
                    st.subheader("🔍 Benzer Araçlar")
                    
                    benzer = df[
                        (df['marka'] == marka) & 
                        (df['seri'] == seri) &
                        (df['yil'].between(yil-2, yil+2))
                    ][['marka', 'seri', 'model', 'yil', 'kilometre_temiz', 'vites_tipi', 'fiyat_temiz']].head(5)
                    
                    if len(benzer) > 0:
                        benzer.columns = ['Marka', 'Seri', 'Model', 'Yıl', 'Kilometre', 'Vites', 'Fiyat (TL)']
                        benzer['Fiyat (TL)'] = benzer['Fiyat (TL)'].apply(lambda x: f"{x:,.0f}")
                        benzer['Kilometre'] = benzer['Kilometre'].apply(lambda x: f"{x:,.0f}")
                        st.dataframe(benzer, use_container_width=True)
                    else:
                        st.warning("Benzer araç bulunamadı.")
                    
                except Exception as e:
                    st.error(f"⚠️ Tahmin yapılırken hata oluştu: {str(e)}")
    
    # ============================================
    # VERİ ANALİZİ SAYFASI
    # ============================================
    elif page == "📊 Veri Analizi":
        st.markdown("---")
        
        # Filtreler
        st.subheader("🔍 Filtreler")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            selected_marka = st.multiselect(
                "Marka",
                options=sorted(df['marka'].unique()),
                default=[]
            )
        
        with col2:
            yil_range = st.slider(
                "Yıl Aralığı",
                min_value=int(df['yil'].min()),
                max_value=int(df['yil'].max()),
                value=(int(df['yil'].min()), int(df['yil'].max()))
            )
        
        with col3:
            fiyat_range = st.slider(
                "Fiyat Aralığı (TL)",
                min_value=int(df['fiyat_temiz'].min()),
                max_value=int(df['fiyat_temiz'].max()),
                value=(int(df['fiyat_temiz'].min()), int(df['fiyat_temiz'].max())),
                step=50000
            )
        
        with col4:
            selected_vites = st.multiselect(
                "Vites Tipi",
                options=sorted(df['vites_tipi'].unique()),
                default=[]
            )
        
        # Filtreleme
        df_filtered = df.copy()
        
        if selected_marka:
            df_filtered = df_filtered[df_filtered['marka'].isin(selected_marka)]
        
        df_filtered = df_filtered[
            (df_filtered['yil'].between(yil_range[0], yil_range[1])) &
            (df_filtered['fiyat_temiz'].between(fiyat_range[0], fiyat_range[1]))
        ]
        
        if selected_vites:
            df_filtered = df_filtered[df_filtered['vites_tipi'].isin(selected_vites)]
        
        st.info(f"📊 Filtrelenmiş kayıt sayısı: {len(df_filtered):,}")
        
        st.markdown("---")
        
        # Grafikler
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dağılımlar", "📈 İlişkiler", "🗺️ Konum", "📋 Veri"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Fiyat Dağılımı")
                fig = px.histogram(
                    df_filtered, 
                    x='fiyat_temiz', 
                    nbins=50,
                    color_discrete_sequence=['#3498db']
                )
                fig.update_layout(
                    xaxis_title="Fiyat (TL)",
                    yaxis_title="Frekans"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🛣️ Kilometre Dağılımı")
                fig = px.histogram(
                    df_filtered, 
                    x='kilometre_temiz', 
                    nbins=50,
                    color_discrete_sequence=['#e74c3c']
                )
                fig.update_layout(
                    xaxis_title="Kilometre",
                    yaxis_title="Frekans"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎨 Boyalı Parça Dağılımı")
                boyali_counts = df_filtered['boyali_sayisi'].value_counts().sort_index()
                fig = px.bar(
                    x=boyali_counts.index,
                    y=boyali_counts.values,
                    color_discrete_sequence=['#f39c12']
                )
                fig.update_layout(
                    xaxis_title="Boyalı Parça Sayısı",
                    yaxis_title="İlan Sayısı"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🔧 Değişen Parça Dağılımı")
                degisen_counts = df_filtered['degisen_sayisi'].value_counts().sort_index()
                fig = px.bar(
                    x=degisen_counts.index,
                    y=degisen_counts.values,
                    color_discrete_sequence=['#9b59b6']
                )
                fig.update_layout(
                    xaxis_title="Değişen Parça Sayısı",
                    yaxis_title="İlan Sayısı"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Yıl - Fiyat İlişkisi")
                yil_fiyat = df_filtered.groupby('yil')['fiyat_temiz'].mean().reset_index()
                fig = px.line(
                    yil_fiyat,
                    x='yil',
                    y='fiyat_temiz',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title="Yıl",
                    yaxis_title="Ortalama Fiyat (TL)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🔄 Kilometre - Fiyat İlişkisi")
                sample = df_filtered.sample(min(1000, len(df_filtered)))
                fig = px.scatter(
                    sample,
                    x='kilometre_temiz',
                    y='fiyat_temiz',
                    color='yil',
                    opacity=0.6
                )
                fig.update_layout(
                    xaxis_title="Kilometre",
                    yaxis_title="Fiyat (TL)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Marka Bazında Fiyat Dağılımı")
            top_markalar = df_filtered['marka'].value_counts().head(10).index
            df_top = df_filtered[df_filtered['marka'].isin(top_markalar)]
            fig = px.box(
                df_top,
                x='marka',
                y='fiyat_temiz',
                color='marka'
            )
            fig.update_layout(
                xaxis_title="Marka",
                yaxis_title="Fiyat (TL)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("📍 İl Bazında İlan Sayısı")
            konum_counts = df_filtered['konum'].value_counts().head(20)
            fig = px.bar(
                x=konum_counts.values,
                y=konum_counts.index,
                orientation='h',
                color=konum_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                xaxis_title="İlan Sayısı",
                yaxis_title="İl",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("💰 İl Bazında Ortalama Fiyat")
            konum_fiyat = df_filtered.groupby('konum')['fiyat_temiz'].mean().sort_values(ascending=False).head(20)
            fig = px.bar(
                x=konum_fiyat.values,
                y=konum_fiyat.index,
                orientation='h',
                color=konum_fiyat.values,
                color_continuous_scale='Greens'
            )
            fig.update_layout(
                xaxis_title="Ortalama Fiyat (TL)",
                yaxis_title="İl",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("📋 Ham Veri")
            
            # Sütun seçimi
            columns_to_show = st.multiselect(
                "Gösterilecek Sütunlar",
                options=df_filtered.columns.tolist(),
                default=['marka', 'seri', 'model', 'yil', 'kilometre_temiz', 'vites_tipi', 'fiyat_temiz']
            )
            
            if columns_to_show:
                st.dataframe(
                    df_filtered[columns_to_show].head(100),
                    use_container_width=True
                )
            
            # İndirme butonu
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 CSV İndir",
                data=csv,
                file_name="araba_verileri.csv",
                mime="text/csv"
            )
    
    # ============================================
    # MODEL PERFORMANSI SAYFASI
    # ============================================
    elif page == "📈 Model Performansı":
        st.markdown("---")
        
        if not model_loaded:
            st.error("⚠️ Model yüklenmedi!")
            return
        
        st.subheader("📊 Model Performans Metrikleri")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="R² Score",
                value=f"{model_info['test_r2']:.4f}",
                delta="Test Seti"
            )
        
        with col2:
            st.metric(
                label="RMSE",
                value=f"{model_info['test_rmse']:,.0f} TL",
                delta="Test Seti"
            )
        
        with col3:
            st.metric(
                label="MAE",
                value=f"{model_info['test_mae']:,.0f} TL",
                delta="Test Seti"
            )
        
        with col4:
            st.metric(
                label="MAPE",
                value=f"{model_info['test_mape']:.2f}%",
                delta="Test Seti"
            )
        
        st.markdown("---")
        
        # Hiperparametreler
        st.subheader("⚙️ En İyi Hiperparametreler")
        
        params_df = pd.DataFrame(
            list(model_info['best_params'].items()),
            columns=['Parametre', 'Değer']
        )
        st.dataframe(params_df, use_container_width=True)
        
        st.markdown("---")
        
        # Özellik önemliliği
        st.subheader("📊 Özellik Önemliliği")
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Özellik': feature_columns,
                'Önem': model.feature_importances_
            }).sort_values('Önem', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Önem',
                y='Özellik',
                orientation='h',
                color='Önem',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_title="Önem Skoru",
                yaxis_title="Özellik",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Metrik açıklamaları
        st.subheader("📚 Metrik Açıklamaları")
        
        st.markdown("""
        | Metrik | Açıklama | İyi Değer |
        |--------|----------|-----------|
        | **R² Score** | Modelin veriyi ne kadar iyi açıkladığını gösterir | > 0.80 |
        | **RMSE** | Ortalama hata miktarı (TL) | Düşük olması iyi |
        | **MAE** | Ortalama mutlak hata (TL) | Düşük olması iyi |
        | **MAPE** | Yüzdelik hata oranı | < %15 |
        """)
    
    # ============================================
    # HAKKINDA SAYFASI
    # ============================================
    elif page == "ℹ️ Hakkında":
        st.markdown("---")
        
        st.subheader("ℹ️ Proje Hakkında")
        
        st.markdown("""
        ### 🚗 Araba Fiyat Tahmini Projesi
        
        Bu proje, makine öğrenmesi kullanarak ikinci el araba fiyatlarını tahmin etmek için geliştirilmiştir.
        
        ---
        
        ### 📊 Veri Seti
        - **Kaynak:** Türkiye'deki ikinci el araba ilanları
        - **Kayıt Sayısı:** ~60,000 ilan
        - **Özellikler:** Marka, seri, model, yıl, kilometre, vites tipi, hasar durumu vb.
        
        ---
        
        ### 🤖 Kullanılan Model
        - **Algoritma:** LightGBM (Light Gradient Boosting Machine)
        - **Optimizasyon:** RandomizedSearchCV ile hiperparametre ayarı
        - **Özellik Mühendisliği:** Araç yaşı, km/yaş oranı, toplam hasar skoru
        
        ---
        
        ### 🛠️ Teknolojiler
        - **Python** - Programlama dili
        - **Pandas & NumPy** - Veri işleme
        - **Scikit-learn** - Makine öğrenmesi
        - **LightGBM** - Gradient boosting
        - **Plotly** - Görselleştirme
        - **Streamlit** - Web arayüzü
        
        ---
        
        ### 👨‍💻 Geliştirici
        Bu proje makine öğrenmesi dersi kapsamında geliştirilmiştir.
        
        ---
        
        ### 📈 Model Performansı
        - Model, test verileri üzerinde yüksek doğruluk oranı göstermektedir
        - Overfitting önlemek için cross-validation kullanılmıştır
        - Hiperparametre optimizasyonu ile en iyi sonuçlar elde edilmiştir
        """)
        
        st.markdown("---")
        
        st.subheader("📁 Proje Yapısı")
        
        st.code("""
        araba_fiyat_tahmini/
        │
        ├── arabalar.csv              # Veri seti
        ├── app.py                    # Streamlit uygulaması
        │
        ├── models/
        │   ├── best_model.pkl        # Eğitilmiş model
        │   ├── scaler.pkl            # Veri ölçekleyici
        │   ├── label_encoders.pkl    # Kategorik encoder'lar
        │   ├── marka_seri_mapping.pkl
        │   ├── feature_columns.pkl
        │   └── model_info.pkl        # Model metrikleri
        │
        └── notebooks/
            └── EDA_and_Model.ipynb   # Analiz notebook'u
        """, language="text")

# ============================================
# UYGULAMAYI ÇALIŞTIR
# ============================================

if __name__ == "__main__":
    main()