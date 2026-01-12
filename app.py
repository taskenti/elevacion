import streamlit as st
import gpxpy
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA "MODERNA Y CAÑERA" ---
st.set_page_config(
    page_title="InstaGPX Profile Generator",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyectar CSS personalizado para look minimalista y fuentes estilo 'Montserrat'
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Montserrat', sans-serif;
        }
        .main {
            background-color: #0e1117;
        }
        h1, h2, h3 {
            color: #ffffff;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-card {
            background-color: #262730;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #363945;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #FF4B4B;
        }
        .metric-label {
            font-size: 12px;
            color: #a0a0a0;
            text-transform: uppercase;
        }
        /* Ajuste fino para gráficos Plotly transparentes */
        .js-plotly-plot .plotly .modebar {
            opacity: 0.5;
        }
    </style>
""", unsafe_allow_html=True)

# --- LÓGICA DE NEGOCIO (Basada en GPX.js e InstaGPX.js) ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula distancia usando radio terrestre de GPX.js (6371.009 km)"""
    R = 6371.009
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def process_gpx_data(file, smoothing_window=1, elevation_threshold=0.5):
    """
    Procesa el GPX aplicando suavizado para mejorar la calidad de los datos
    (como solicita el usuario al decir 'los datos no son tan buenos').
    """
    gpx = gpxpy.parse(file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation,
                    'time': point.time
                })
    
    if not points:
        return None, None

    df = pd.DataFrame(points)

    # 1. Suavizado de Elevación (Técnica clave para "Buenos Datos")
    # Los GPX crudos tienen mucho ruido. Aplicamos una media móvil.
    if smoothing_window > 1:
        df['ele'] = df['ele'].rolling(window=smoothing_window, center=True, min_periods=1).mean()

    # 2. Cálculos acumulativos
    distances = [0]
    gain = 0
    loss = 0
    
    # Iteración optimizada con numpy/pandas vectorizado sería ideal, 
    # pero el bucle explícito permite aplicar el "Umbral" (hysteresis) fácilmente
    for i in range(1, len(df)):
        d = haversine_distance(df.loc[i-1, 'lat'], df.loc[i-1, 'lon'], 
                               df.loc[i, 'lat'], df.loc[i, 'lon'])
        distances.append(distances[-1] + d)
        
        diff = df.loc[i, 'ele'] - df.loc[i-1, 'ele']
        
        # Filtro de umbral para evitar sumar ruido como desnivel
        if abs(diff) >= elevation_threshold:
            if diff > 0:
                gain += diff
            else:
                loss += diff

    df['dist_km'] = distances
    
    stats = {
        'distance': distances[-1],
        'gain': gain,
        'loss': loss,
        'max_ele': df['ele'].max(),
        'min_ele': df['ele'].min(),
        'avg_ele': df['ele'].mean()
    }
    
    return df, stats

# --- INTERFAZ DE USUARIO ---

st.title("🏔️ InstaGPX PRO")
st.markdown("Generador de perfiles de altimetría de alta precisión y estética.")

# Sidebar de Configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    uploaded_file = st.file_uploader("📂 Subir archivo GPX", type=["gpx"])

    with st.expander("🛠️ Procesamiento de Datos", expanded=True):
        st.markdown("**Mejora la calidad de tus datos**")
        smoothing = st.slider("Suavizado (Rolling Window)", 1, 50, 5, 
                             help="Reduce el ruido del GPS promediando puntos vecinos.")
        threshold = st.slider("Umbral de Desnivel (m)", 0.0, 5.0, 1.0, 0.5,
                             help="Ignora cambios de altura menores a este valor para no inflar el desnivel acumulado.")

    st.header("🎨 Diseño Gráfico")
    
    # Las 8 opciones solicitadas
    line_color = st.color_picker("1. Color de Línea", "#FFFFFF")
    fill_color = st.color_picker("2. Color de Relleno", "#FFFFFF") # Se aplicará transparencia luego
    fill_opacity = st.slider("3. Opacidad de Relleno", 0.0, 1.0, 0.25)
    line_width = st.slider("4. Grosor de Línea", 1, 10, 4)
    show_grid_x = st.checkbox("5. Grid Vertical (10km)", value=True)
    show_grid_y = st.checkbox("6. Grid Horizontal", value=True)
    grid_opacity = st.slider("7. Opacidad del Grid", 0.0, 1.0, 0.1)
    bg_style = st.selectbox("8. Estilo de Fondo", ["Transparente (InstaGPX)", "Oscuro", "Claro"])

if uploaded_file:
    with st.spinner("Procesando ruta con algoritmos de corrección..."):
        df, stats = process_gpx_data(uploaded_file, smoothing, threshold)

    if df is not None:
        
        # --- SECCIÓN DE MÉTRICAS (Estilo Minimalista) ---
        c1, c2, c3, c4 = st.columns(4)
        
        def metric_html(label, value):
            return f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """
            
        c1.markdown(metric_html("Distancia", f"{stats['distance']:.2f} km"), unsafe_allow_html=True)
        c2.markdown(metric_html("Desnivel +", f"{int(stats['gain'])} m"), unsafe_allow_html=True)
        c3.markdown(metric_html("Desnivel -", f"{int(abs(stats['loss']))} m"), unsafe_allow_html=True)
        c4.markdown(metric_html("Altitud Max", f"{int(stats['max_ele'])} m"), unsafe_allow_html=True)
        
        st.markdown("---")

        # --- GENERACIÓN DEL GRÁFICO (Estilo InstaGPX.js) ---
        
        # Convertir colores hex a rgba para Plotly
        def hex_to_rgba(hex_color, opacity):
            hex_color = hex_color.lstrip('#')
            return f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {opacity})"

        fill_rgba = hex_to_rgba(fill_color, fill_opacity)
        grid_rgba = f"rgba(255, 255, 255, {grid_opacity})" if bg_style != "Claro" else f"rgba(0, 0, 0, {grid_opacity})"
        
        # Configuración de Fondo
        paper_bg = "#000000" if bg_style == "Oscuro" else "#FFFFFF" if bg_style == "Claro" else "rgba(0,0,0,0)"
        plot_bg = "rgba(255,255,255,0.05)" if bg_style == "Transparente (InstaGPX)" else paper_bg

        fig = go.Figure()

        # Área rellena
        fig.add_trace(go.Scatter(
            x=df['dist_km'],
            y=df['ele'],
            mode='lines',
            line=dict(color=line_color, width=line_width),
            fill='tozeroy',
            fillcolor=fill_rgba,
            name='Elevación',
            hoverinfo='x+y'
        ))

        # Replicar lógica de grid de InstaGPX.js:
        # "let _blockKms = Math.ceil(Math.ceil((distance.km)/10));"
        # Esto sugiere un grid cada 10km o dinámico si es muy corto.
        total_dist = stats['distance']
        tick_vals = []
        if show_grid_x:
            if total_dist > 20:
                step = 10
            elif total_dist > 10:
                step = 5
            else:
                step = 1
            tick_vals = np.arange(0, total_dist + step, step)

        # Configuración del Layout para imitar Canvas roundRect y estilos
        fig.update_layout(
            title="",
            autosize=True,
            height=500,
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            margin=dict(l=20, r=20, t=30, b=30),
            
            xaxis=dict(
                showgrid=show_grid_x,
                gridcolor=grid_rgba,
                gridwidth=1, # InstaGPX usa 2px pero en plotly 1 queda mas fino
                tickmode='array' if show_grid_x else 'auto',
                tickvals=tick_vals if show_grid_x else None,
                zeroline=False,
                showticklabels=True,
                tickfont=dict(color="#888", family="Montserrat")
            ),
            yaxis=dict(
                showgrid=show_grid_y,
                gridcolor=grid_rgba,
                zeroline=False,
                showticklabels=True,
                tickfont=dict(color="#888", family="Montserrat")
                # En InstaGPX calculan "_altitudeAxisNum" (1-6 líneas). Plotly lo hace auto.
            ),
            hovermode="x unified"
        )
        
        # Mostrar gráfica
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Perfil generado usando `{len(df)}` puntos trackeados. Ventana de suavizado: `{smoothing}`.")

    else:
        st.error("No se pudieron extraer puntos válidos del archivo GPX.")
else:
    # Pantalla de bienvenida / Empty State
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #666;">
        <h3>👋 Sube tu ruta para comenzar</h3>
        <p>Analiza el perfil, optimiza los datos y exporta tu gráfico.</p>
    </div>
    """, unsafe_allow_html=True)
