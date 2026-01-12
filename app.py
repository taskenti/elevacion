import streamlit as st
import gpxpy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Perfil de Altimetría GPX", page_icon="🏔️", layout="wide")

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Replica la función 'pointsDistance' del archivo GPX.js original.
    Calcula la distancia entre dos puntos geográficos.
    """
    R = 6371.009  # Radio de la tierra en KM (mismo valor que en GPX.js)
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    d = R * c
    return d # Retorna distancia en KM

def process_gpx(file):
    """
    Procesa el archivo GPX extrayendo lat, lon, elevación y calculando distancias
    basado en la lógica de 'elevation' IIFE en GPX.js.
    """
    gpx = gpxpy.parse(file)
    
    data = []
    cumulative_distance = 0
    elevation_gain = 0
    elevation_loss = 0
    
    # Extraer todos los puntos en una lista plana
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(point)
    
    if not points:
        return None, None

    # Procesar puntos (lógica similar al bucle for en GPX.js)
    start_time = points[0].time
    
    for i in range(len(points)):
        p = points[i]
        ele = p.elevation
        
        # Calcular distancia y desnivel respecto al punto anterior
        dist_segment = 0
        if i > 0:
            prev_p = points[i-1]
            dist_segment = haversine_distance(prev_p.latitude, prev_p.longitude, p.latitude, p.longitude)
            cumulative_distance += dist_segment
            
            ele_diff = p.elevation - prev_p.elevation
            if ele_diff > 0:
                elevation_gain += ele_diff
            elif ele_diff < 0:
                elevation_loss += ele_diff

        data.append({
            "lat": p.latitude,
            "lon": p.longitude,
            "elevation": p.elevation,
            "distance_km": cumulative_distance,
            "time": p.time
        })

    df = pd.DataFrame(data)
    
    metrics = {
        "total_distance_km": cumulative_distance,
        "max_elevation": df["elevation"].max(),
        "min_elevation": df["elevation"].min(),
        "gain": elevation_gain,
        "loss": elevation_loss
    }
    
    return df, metrics

# --- Interfaz de Usuario ---

st.title("🏔️ Generador de Perfil de Altimetría")
st.markdown("""
Esta aplicación genera un perfil de elevación detallado a partir de archivos GPX, 
utilizando algoritmos de cálculo de distancia geodésica.
""")

uploaded_file = st.file_uploader("Sube tu archivo .GPX", type=["gpx"])

if uploaded_file is not None:
    try:
        with st.spinner('Procesando ruta...'):
            df, metrics = process_gpx(uploaded_file)
            
        if df is not None:
            # 1. Mostrar Métricas Principales
            st.subheader("📊 Datos de la Ruta")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Distancia Total", f"{metrics['total_distance_km']:.2f} km")
            col2.metric("Desnivel Positivo (+)", f"{metrics['gain']:.0f} m")
            col3.metric("Desnivel Negativo (-)", f"{abs(metrics['loss']):.0f} m")
            col4.metric("Altitud Máxima", f"{metrics['max_elevation']:.0f} m")

            # 2. Gráfico de Perfil de Altimetría (Plotly)
            st.subheader("📈 Perfil de Elevación")
            
            fig = px.area(
                df, 
                x="distance_km", 
                y="elevation", 
                title="Perfil de Altimetría (Distancia vs Elevación)",
                labels={"distance_km": "Distancia (km)", "elevation": "Elevación (m)"},
                color_discrete_sequence=["#FF4B4B"] # Color estilo Streamlit/Strava
            )
            
            # Mejorar el diseño del gráfico
            fig.update_layout(
                hovermode="x unified",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                plot_bgcolor="white"
            )
            
            # Añadir línea de altitud mínima para que el área no empiece en 0 si estamos a 2000m
            min_ele = metrics['min_elevation'] * 0.9
            fig.update_yaxes(range=[min_ele, metrics['max_elevation'] * 1.1])
            
            st.plotly_chart(fig, use_container_width=True)

            # 3. Mapa de la Ruta
            st.subheader("🗺️ Mapa del Recorrido")
            
            # Usamos Scatter Mapbox para una visualización rápida
            map_fig = px.line_mapbox(
                df, 
                lat="lat", 
                lon="lon", 
                hover_name="elevation",
                zoom=10, 
                height=500
            )
            
            map_fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            st.plotly_chart(map_fig, use_container_width=True)

            # 4. Tabla de datos (Opcional)
            with st.expander("Ver datos crudos"):
                st.dataframe(df)
                
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

else:
    st.info("Por favor, sube un archivo GPX para comenzar.")
