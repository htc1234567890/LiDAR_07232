import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Home")

st.sidebar.success("Select a tool above.")

st.markdown(
    """
    Welcome to the Lidar Processing Toolkit!

    This application provides tools for visualizing and processing 3D point cloud data.

    **👈 Select a tool from the sidebar** to get started.

    ### Available Tools:
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Background Filtering")
    st.write("Build a background model and filter dynamic objects.")
    if st.button("Go to Background Filtering", use_container_width=True):
        st.switch_page("pages/1_Background_Filtering.py")

with col2:
    st.subheader("Object Detection and Tracking")
    st.write("Detect and track objects in filtered point clouds.")
    if st.button("Go to Object Detection and Tracking", use_container_width=True):
        st.switch_page("pages/2_Object_Detection_and_Tracking.py")



