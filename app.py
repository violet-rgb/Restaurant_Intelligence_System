import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Restaurant Intelligence System",
    layout="wide"
)

# =====================================================
# CUSTOM THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #111111;
}

h1, h2, h3 {
    color: #FF4B4B !important;
}

[data-testid="metric-container"] {
    background-color: #1A1A1A;
    border: 1px solid #FF4B4B;
    border-radius: 12px;
    padding: 15px;
}

.stButton > button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
}

.stButton > button:hover {
    background-color: #D62828;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "data/cleaned_restaurant_dataset.csv"
)

model = joblib.load(
    "models/restaurant_rating_model.pkl"
)

# =====================================================
# CHART THEME FUNCTION
# =====================================================


def apply_theme(fig):

    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        title_font_color="#FF4B4B"
    )

    return fig

# =====================================================
# SIDEBAR
# =====================================================


page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Analysis",
        "Prediction"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🍽 Restaurant Intelligence System")

    st.markdown("""
    Discover restaurant trends, customer preferences,
    pricing patterns and restaurant ratings.
    """)

    # ---------------- Filters ----------------

    country = st.sidebar.selectbox(
        "Country",
        ["All"] +
        sorted(
            df["Country"]
            .dropna()
            .unique()
        )
    )

    if country != "All":
        filtered_df = df[
            df["Country"] == country
        ]
    else:
        filtered_df = df

    # ---------------- KPI CARDS ----------------

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Restaurants",
        len(filtered_df)
    )

    c2.metric(
        "Countries",
        filtered_df["Country"].nunique()
    )

    c3.metric(
        "Cities",
        filtered_df["City"].nunique()
    )

    c4.metric(
        "Avg Rating",
        round(
            filtered_df["Aggregate rating"].mean(),
            2
        )
    )

    c5.metric(
        "Top City",
        filtered_df["City"]
        .value_counts()
        .idxmax()
    )

    c6.metric(
        "Top Price",
        filtered_df["Price range"]
        .mode()[0]
    )

    st.divider()

    # ---------------- Dataset ----------------

    st.subheader("Dataset Preview")

    st.dataframe(
        filtered_df.head()
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        filtered_df.describe()
    )

    # ---------------- MAP ----------------

    st.subheader("Restaurant Locations")

    map_df = filtered_df.dropna(
        subset=[
            "Latitude",
            "Longitude"
        ]
    )

    fig_map = px.scatter_map(
        map_df,
        lat="Latitude",
        lon="Longitude",
        color="Aggregate rating",
        hover_data=[
            "City",
            "Country"
        ],
        zoom=3,
        height=500
    )

    fig_map = apply_theme(fig_map)

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

    # ---------------- TOP CITIES ----------------

    city_counts = (
        filtered_df["City"]
        .value_counts()
        .head(10)
    )

    fig1 = px.bar(
        x=city_counts.index,
        y=city_counts.values,
        title="Top 10 Cities"
    )

    fig1 = apply_theme(fig1)

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # ---------------- TOP CUISINES ----------------

    cuisines = (
        filtered_df["Cuisines"]
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(10)
    )

    fig2 = px.bar(
        x=cuisines.index,
        y=cuisines.values,
        title="Top 10 Cuisines"
    )

    fig2 = apply_theme(fig2)

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ---------------- PRICE RANGE ----------------

    price_counts = (
        filtered_df["Price range"]
        .value_counts()
    )

    fig3 = px.pie(
        values=price_counts.values,
        names=price_counts.index,
        title="Price Range Distribution"
    )

    fig3 = apply_theme(fig3)

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =====================================================
# ANALYSIS PAGE
# =====================================================

elif page == "Analysis":

    st.title("📊 Analysis & Insights")

    st.info(
        "Key findings from customer preference analysis."
    )

    st.write(
        "✔ North Indian cuisine is the most common cuisine."
    )

    st.write(
        "✔ New Delhi contains the highest number of restaurants."
    )

    st.write(
        "✔ Restaurants with table booking receive higher ratings."
    )

    st.write(
        "✔ Higher price ranges generally receive higher ratings."
    )

    st.write(
        "✔ Votes are the strongest predictor of ratings."
    )

    importance_df = pd.DataFrame({
        "Feature": [
            "Votes_Log",
            "Average Cost",
            "Price Range",
            "Online Delivery",
            "Address Length"
        ],
        "Importance": [
            0.819,
            0.101,
            0.041,
            0.022,
            0.016
        ]
    })

    fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        title="Feature Importance"
    )

    fig = apply_theme(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.title("🤖 Restaurant Rating Predictor")

    cost = st.slider(
        "Average Cost for Two",
        0,
        5000,
        500,
        100
    )

    price_range_text = st.select_slider(
        "Price Range",
        options=[
            "Budget",
            "Moderate",
            "Premium",
            "Luxury"
        ]
    )

    price_map = {
        "Budget": 1,
        "Moderate": 2,
        "Premium": 3,
        "Luxury": 4
    }

    price_range = price_map[
        price_range_text
    ]

    votes = st.slider(
        "Customer Votes",
        0,
        10000,
        100
    )

    table_booking = st.toggle(
        "Table Booking"
    )

    online_delivery = st.toggle(
        "Online Delivery"
    )

    restaurant_name_length = st.slider(
        "Restaurant Name Length",
        1,
        50,
        15
    )

    address_length = st.slider(
        "Address Length",
        1,
        150,
        50
    )

    cuisine_count = st.slider(
        "Cuisine Count",
        1,
        10,
        2
    )

    if st.button("Predict Rating"):

        input_data = np.array([
            [
                cost,
                price_range,
                np.log1p(votes),
                int(table_booking),
                int(online_delivery),
                restaurant_name_length,
                address_length,
                cuisine_count
            ]
        ])

        prediction = model.predict(
            input_data
        )[0]

        st.success(
            f"⭐ Predicted Rating: {prediction:.2f}"
        )

        if prediction >= 4.5:
            st.success("🌟 Excellent Restaurant")

        elif prediction >= 4:
            st.success("⭐ Very Good Restaurant")

        elif prediction >= 3:
            st.warning("👍 Average Restaurant")

        else:
            st.error("⚠ Needs Improvement")
