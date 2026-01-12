import streamlit as st
import gpxpy
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import io

# --- CONFIGURACIÓN DE PÁGINA ESTILO KOMOOT ---
st.set_page_config(
    page_title="Altimetría Pro",
    page_icon="⛰️",
    layout="centered", # Komoot usa contenedores centrados para enfoque
    initial_sidebar_state="collapsed"
)

# Inyectar CSS para imitar la UI limpia de Komoot
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Open Sans', sans-serif;
            color: #333333;
        }
        .stApp {
            background-color: #f8f9fa; /* Gris muy suave, típico de apps modernas */
        }
        h1 {
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #2c3e50;
        }
        h3 {
            font-weight: 600;
            color: #566573;
            font-size: 1.1rem !important;
            margin-top: 20px !important;
        }
        .metric-container {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .metric-val {
            font-size: 22px;
            font-weight: 700;
            color: #2c3e50;
        }
        .metric-lbl {
            font-size: 11px;
            color: #8899a6;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }
        /* Botones personalizados */
        .stButton button {
            border-radius: 6px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# --- LÓGICA DE PROCESAMIENTO (Igual que antes, robusta) ---

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.009
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def process_gpx(file, smoothing=5):
    gpx = gpxpy.parse(file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({'lat': point.latitude, 'lon': point.longitude, 'ele': point.elevation})
    
    if not points: return None, None
    
    df = pd.DataFrame(points)
    
    # Suavizado inteligente (clave para calidad "Komoot")
    if smoothing > 1:
        df['ele'] = df['ele'].rolling(window=smoothing, center=True, min_periods=1).mean()
        
    distances = [0]
    gain, loss = 0, 0
    
    for i in range(1, len(df)):
        d = haversine_distance(df.loc[i-1,'lat'], df.loc[i-1,'lon'], df.loc[i,'lat'], df.loc[i,'lon'])
        distances.append(distances[-1] + d)
        diff = df.loc[i, 'ele'] - df.loc[i-1, 'ele']
        if diff > 0.5: gain += diff
        elif diff < -0.5: loss += diff
            
    df['dist_km'] = distances
    
    # Calcular rangos para ajuste perfecto de ejes
    ele_min = df['ele'].min()
    ele_max = df['ele'].max()
    ele_buffer = (ele_max - ele_min) * 0.1 # 10% de margen
    
    stats = {
        'dist': distances[-1],
        'gain': gain,
        'loss': loss,
        'min': ele_min,
        'max': ele_max,
        'y_range': [ele_min - ele_buffer, ele_max + ele_buffer]
    }
    return df, stats

# --- UI PRINCIPAL ---

st.title("Generador de Perfil GPX")
st.markdown("Crea perfiles de altimetría con calidad de publicación.")

col_upload, col_conf = st.columns([1, 2])
with col_upload:
    uploaded_file = st.file_uploader("Sube archivo GPX", type=["gpx"])

# Configuración de diseño (Sidebar o Inline)
with st.expander("🎨 Personalizar Diseño del Gráfico", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        line_color = st.color_picker("Color Línea", "#36454F") # Charcoal Komoot
    with c2:
        fill_color = st.color_picker("Color Relleno", "#85C1E9") # Azul suave
    with c3:
        fill_opacity = st.slider("Opacidad", 0.0, 1.0, 0.3)
    
    st.caption("Tip: Para estilo Komoot, usa líneas oscuras y rellenos suaves o grises.")

if uploaded_file:
    df, stats = process_gpx(uploaded_file)
    
    if df is not None:
        # 1. Métricas Estilo Tarjeta
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        
        def card(label, val):
            return f"""<div class="metric-container"><div class="metric-val">{val}</div><div class="metric-lbl">{label}</div></div>"""
            
        m1.markdown(card("Distancia", f"{stats['dist']:.1f} <span style='font-size:12px'>km</span>"), unsafe_allow_html=True)
        m2.markdown(card("Desnivel +", f"{int(stats['gain'])} <span style='font-size:12px'>m</span>"), unsafe_allow_html=True)
        m3.markdown(card("Desnivel -", f"{int(abs(stats['loss']))} <span style='font-size:12px'>m</span>"), unsafe_allow_html=True)
        m4.markdown(card("Altitud Max", f"{int(stats['max'])} <span style='font-size:12px'>m</span>"), unsafe_allow_html=True)

        # 2. Generación del Gráfico Plotly
        st.markdown("### Perfil de Elevación")
        
        # Convertir hex a rgba
        hex_c = fill_color.lstrip('#')
        rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{fill_opacity})"
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['dist_km'], y=df['ele'],
            mode='lines',
            line=dict(color=line_color, width=2.5),
            fill='tozeroy',
            fillcolor=fill_rgba,
            hoverinfo='x+y',
            hovertemplate='%{y:.0f}m<br>%{x:.1f}km<extra></extra>' # Tooltip limpio
        ))
        
        # Layout profesional "Komoot Style"
        fig.update_layout(
            autosize=True,
            height=400,
            margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            
            xaxis=dict(
                title="Distancia (km)",
                title_font=dict(size=12, color='#888'),
                showgrid=True,
                gridcolor='#eee',
                gridwidth=1,
                zeroline=False,
                tickfont=dict(color='#666', size=11),
                fixedrange=False # Permite zoom usuario
            ),
            
            yaxis=dict(
                title="Altitud (m)",
                title_font=dict(size=12, color='#888'),
                showgrid=True,
                gridcolor='#eee',
                gridwidth=1,
                zeroline=False,
                tickfont=dict(color='#666', size=11),
                range=stats['y_range'], # AJUSTE INTELIGENTE DE EJES
                fixedrange=False
            ),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 3. Zona de Descargas
        st.markdown("### Exportar Gráfico")
        st.markdown("Descarga el perfil en alta calidad PNG para compartir en redes sociales.")
        
        d1, d2, d3 = st.columns(3)
        
        # Función auxiliar para convertir a bytes
        def to_png_bytes(figure, transparent=False, bg_color="white"):
            # Configuramos el fondo para la descarga
            img_fig = go.Figure(figure) # Copia para no alterar la vista
            if not transparent:
                img_fig.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)
            else:
                img_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                
            return img_fig.to_image(format="png", width=1200, height=600, scale=2)

        try:
            # Opción 1: Transparente (Ideal overlays)
            btn_trans = d1.download_button(
                label="🖼️ PNG Transparente",
                data=to_png_bytes(fig, transparent=True),
                file_name="perfil_transparente.png",
                mime="image/png"
            )
            
            # Opción 2: Fondo Blanco (Clásico Komoot Web)
            btn_white = d2.download_button(
                label="📄 PNG Fondo Blanco",
                data=to_png_bytes(fig, transparent=False, bg_color="white"),
                file_name="perfil_blanco.png",
                mime="image/png"
            )
            
            # Opción 3: Fondo Oscuro (Modo noche)
            btn_dark = d3.download_button(
                label="🌑 PNG Fondo Oscuro",
                data=to_png_bytes(fig, transparent=False, bg_color="#1e1e1e"),
                file_name="perfil_oscuro.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.warning("⚠️ Para habilitar la descarga de imágenes estáticas, instala 'kaleido': `pip install kaleido`")
            st.error(f"Error técnico: {e}")

    else:
        st.error("Archivo GPX inválido o vacío.")

else:
    # Empty State elegante
    st.info("👆 Sube un archivo .GPX para ver la magia.")
