import streamlit as st
import gpxpy
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- 1. CONFIGURACIÓN DE PÁGINA (WIDE MODE PARA MAPAS/GRÁFICOS) ---
st.set_page_config(
    page_title="Ruta GPX",
    page_icon="🚲",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. ESTILOS CSS AVANZADOS (KOMOOT CLONE) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
        
        /* Reset general */
        html, body, [class*="css"]  {
            font-family: 'Open Sans', sans-serif;
            background-color: #ffffff; /* Fondo blanco puro Komoot */
            color: #333333;
        }
        
        /* Ocultar elementos nativos de Streamlit para limpiar la UI */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Título Principal */
        .main-title {
            font-size: 24px;
            font-weight: 700;
            color: #262626;
            margin-bottom: 5px;
        }
        
        /* Contenedor de Métricas (Flexbox simple) */
        .stats-container {
            display: flex;
            justify-content: space-between;
            padding: 20px 0;
            border-bottom: 1px solid #e6e6e6;
            margin-bottom: 20px;
        }
        
        .stat-box {
            text-align: left;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: #262626;
            line-height: 1.2;
        }
        
        .stat-label {
            font-size: 13px;
            color: #757575; /* Gris medio Komoot */
            font-weight: 400;
        }
        
        .unit {
            font-size: 14px;
            font-weight: 600;
            color: #555;
        }

        /* Ajustes del contenedor del gráfico */
        .js-plotly-plot {
            border-radius: 0px;
        }
        
        /* Botón de carga personalizado */
        .stFileUploader label {
            font-size: 14px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. LÓGICA DE PROCESAMIENTO ROBUSTA ---

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.009
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def process_gpx(file):
    try:
        gpx = gpxpy.parse(file)
    except Exception:
        return None, None

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'lat': point.latitude, 
                    'lon': point.longitude, 
                    'ele': point.elevation
                })
    
    if not points: return None, None
    
    df = pd.DataFrame(points)
    
    # SUAVIZADO: Komoot suaviza mucho para evitar ruido
    # Ventana de 15 puntos para una curva muy limpia
    df['ele_smooth'] = df['ele'].rolling(window=15, center=True, min_periods=1).mean()
        
    distances = [0]
    gain, loss = 0, 0
    
    for i in range(1, len(df)):
        d = haversine_distance(df.loc[i-1,'lat'], df.loc[i-1,'lon'], df.loc[i,'lat'], df.loc[i,'lon'])
        distances.append(distances[-1] + d)
        
        # Calcular desnivel sobre la elevación suavizada (más realista)
        diff = df.loc[i, 'ele_smooth'] - df.loc[i-1, 'ele_smooth']
        if diff > 0: gain += diff
        else: loss += diff
            
    df['dist_km'] = distances
    
    # Estadísticas
    stats = {
        'dist': distances[-1],
        'gain': gain,
        'loss': loss,
        'min_ele': df['ele_smooth'].min(),
        'max_ele': df['ele_smooth'].max()
    }
    
    # Rango Y inteligente (buffer arriba y abajo)
    ele_range = stats['max_ele'] - stats['min_ele']
    stats['y_min'] = stats['min_ele'] - (ele_range * 0.05)
    stats['y_max'] = stats['max_ele'] + (ele_range * 0.1) # Un poco más de aire arriba
    
    return df, stats

# --- 4. INTERFAZ DE USUARIO ---

st.markdown('<div class="main-title">Perfil de Elevación</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["gpx"], help="Arrastra tu archivo GPX aquí")

if uploaded_file:
    df, stats = process_gpx(uploaded_file)
    
    if df is not None:
        
        # --- A. MÉTRICAS ESTILO KOMOOT (HTML Puro) ---
        # Usamos HTML para controlar exactamente el espaciado y la fuente
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-box">
                <div class="stat-label">Distancia</div>
                <div class="stat-value">{stats['dist']:.1f} <span class="unit">km</span></div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Desnivel positivo</div>
                <div class="stat-value">{int(stats['gain'])} <span class="unit">m</span></div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Desnivel negativo</div>
                <div class="stat-value">{int(abs(stats['loss']))} <span class="unit">m</span></div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Altitud máx</div>
                <div class="stat-value">{int(stats['max_ele'])} <span class="unit">m</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- B. GRÁFICO EXACTO ---
        
        fig = go.Figure()
        
        # Línea y Relleno
        fig.add_trace(go.Scatter(
            x=df['dist_km'], 
            y=df['ele_smooth'],
            mode='lines',
            line=dict(
                color='#333333', # Gris casi negro (estándar navegación)
                width=1.5        # Línea fina y elegante
            ),
            fill='tozeroy',      # Rellenar hasta el eje 0 (luego recortamos con el eje Y)
            fillcolor='rgba(51, 51, 51, 0.1)', # Gris muy suave transparente
            hoverinfo='x+y',
            hovertemplate='<b>%{y:.0f} m</b><br>%{x:.1f} km<extra></extra>'
        ))
        
        # Layout Minimalista
        fig.update_layout(
            autosize=True,
            height=350, # Altura compacta típica de perfiles web
            margin=dict(l=0, r=0, t=10, b=0), # Sin márgenes externos
            plot_bgcolor='white',
            paper_bgcolor='white',
            
            # Eje X (Distancia)
            xaxis=dict(
                showgrid=False,      # Komoot no suele tener grid vertical fuerte
                zeroline=False,
                showline=True,       # Línea base sutil
                linecolor='#e0e0e0',
                tickfont=dict(size=11, color='#757575', family='Open Sans'),
                ticks="outside",
                ticklen=5,
                title="",            # Sin título para limpieza (se entiende por contexto)
                fixedrange=True      # Fijo para evitar zooms accidentales si se quiere estático
            ),
            
            # Eje Y (Elevación)
            yaxis=dict(
                showgrid=True,       # Grid horizontal SÍ (guía visual de altura)
                gridcolor='#f0f0f0', # Grid muy sutil
                gridwidth=1,
                zeroline=False,
                showline=False,      # Sin línea vertical izquierda
                tickfont=dict(size=11, color='#757575', family='Open Sans'),
                range=[stats['y_min'], stats['y_max']], # Zoom inteligente
                side='right',        # Etiquetas a la derecha (estilo moderno)
                fixedrange=True
            ),
            hovermode="x unified",
            showlegend=False
        )
        
        # Renderizar
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
            'toImageButtonOptions': {
                'format': 'png', # Uno de png, svg, jpeg, webp
                'filename': 'mi_perfil_gpx',
                'height': 500,
                'width': 1000,
                'scale': 2 # Alta resolución
            }
        })
        
        st.caption("💡 Pasa el ratón por el gráfico y haz clic en el icono de la cámara (arriba a la derecha) para descargar la imagen en Alta Resolución.")

    else:
        st.error("El archivo no contiene datos de elevación válidos.")

else:
    # Empty state minimalista
    st.markdown("""
    <div style="background-color: #f9f9f9; padding: 40px; border-radius: 8px; text-align: center; border: 1px dashed #ccc;">
        <h4 style="color: #555; margin:0;">Sube un archivo GPX para comenzar</h4>
        <p style="color: #999; font-size: 14px;">Se generará el perfil de altimetría automáticamente.</p>
    </div>
    """, unsafe_allow_html=True)
