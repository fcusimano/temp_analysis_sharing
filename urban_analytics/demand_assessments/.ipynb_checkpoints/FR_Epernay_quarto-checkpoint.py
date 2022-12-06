# ---
# title: 'Épernay'
# date: today
# categories: 'France'
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from urban_analytics.utils import *
from urban_analytics.imports import *
from urban_analytics.geography import *
from urban_analytics.fleet_sizing import *
from urban_analytics.h3_utils import *
from urban_analytics.osm import *
from urban_analytics.scoring import *
from urban_analytics.population import *
from urban_analytics.station_selection import *
from urban_analytics.maps import *
from urban_analytics.predictions import *

import fiona
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from IPython.display import Markdown
from dash import Dash, html, dcc
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output

init_notebook_mode(connected=True)
plotly.offline.init_notebook_mode(connected=True)
from matplotlib.lines import Line2D
import contextily as cx

# %load_ext lab_black

# %%
place = "Épernay"
country_code = "FR"

local_crs = LocalCRS[country_code].value
data_dir_in, data_dir_out = set_up_directories(country_code, place)

# %%
# | output: true
if country_code == "FR":
    display(Markdown("# Étude de territoire\n\n"))
else:
    display(Markdown("# Demand Assessment\n\n"))

# %%
display(
    Markdown(
        f"""
## Fleet sizing
### VLS market
"""
    )
)

# %%
target_cities_df = get_city_selection_df(
    target_city_name="epernay",
    city_subset=["epernay", "Magenta", "Moussy", "Pierry", "Vinay", "Mardeuil"],
)

# %%
market_vls = get_vls_fleet(target_cities_df, batches=10)
market_vls.reset_index(inplace=True)
market_vls.rename(columns={"index": "CODGEO"}, inplace=True)
market_vls.sort_values("rounded_bikes", ascending=False).head(20)

# %%
market_vls["stations"] = 1
market_vls["stations"][0] = 14
market_vls["rounded_bikes"] = 5
market_vls["rounded_bikes"][0] = 80
market_vls

# %%
display(
    Markdown(
        f"""
### VLD market
"""
    )
)

# %%
target_cities_df_VLD = target_cities_df.copy()
target_cities_df_VLD.rename(
    columns={"LIBGEO": "city_name", "DEP": "dept_id"}, inplace=True
)

# %%
market_vld = get_vld_fleet(target_cities_df_VLD, batches=5)
print("Recommended long bike rental fleet size by city:")
print(
    f"{len(market_vld)} cities and {int(market_vld.bikes_lower.sum())}-{int(market_vld.bikes_upper.sum())} bikes"
)
print_fleet_df(
    market_vld[
        [
            "city",
            "bikes",
        ]
    ]
)
vld_bikes = market_vld["bikes"].sum()

# %%
display(Markdown(f"## Geography"))

# %%
specific_place_name = get_specific_place_name(target_cities_df)
city = get_city(specific_place_name, local_crs)

# %%
city.plot(
    column="city_name",
    legend=False,
    figsize=(6, 8),
    legend_kwds={"bbox_to_anchor": (1.5, 1)},
).axis("off")

# %%
display(Markdown(f"### H3"))

# %%
hex_resolution = 10

# Generate H3 hexagons
base = get_h3_fill(city, hex_resolution, local_crs)

# Plot
ax = base.plot(figsize=(8, 8))
city.plot(ax=ax, color="None").axis("off")

# %%
display(Markdown(f"### External data"))

# %%
ox.settings.timeout = 1000

# %%
geo_data_dict = {}
geo_data_dict.update(get_lanes_data(specific_place_name))
geo_data_dict.update(get_bike_parking_data(specific_place_name))
geo_data_dict.update(get_transit_data(specific_place_name, country_code))
geo_data_dict.update(get_offices_data(specific_place_name))
geo_data_dict.update(get_destinations_data(specific_place_name, country_code))
geo_data_dict.update(get_population_data(city, data_dir_out, place, country_code))

# %%
to_remove_tuple = [
    # (
    #    "DESTINATIONS",
    #    "hospitals",
    #    ["Hôpital de la Cavale Blanche"],
    # )   ,
    # (
    #    "DESTINATIONS",
    #    "townhalls",
    #    [
    #        "Mairie de Bondoufle",
    #        "Centre administratif Darblay",
    #        "Mairie de Juvisy-sur-Orge",
    #        'Résidence "Collette"',
    #    ],
    # ),
    # (
    #    "DESTINATIONS",
    #    "universities",
    #    [
    #        "Lycée Montmajour et Perdiguier",
    #        "École Nationale Supérieure de la Photographie",
    #    ],
    # ),
    # (
    #    "TRANSIT",
    #    "train_stations",
    #    ["Breuillet - Village", "Corbeil-Essonnes", "Athis-Mons", "Savigny-sur-Orge"],
    # ),
    # (
    #    "TRANSIT",
    #    "bus_stations",
    #    [
    #        "Saint-Michel-sur-Orge RER",
    #        "Gare de Sainte-Geneviève des Bois RER C",
    #        "Évry-Courcouronnes Centre",
    #        "Gare RER C de Savigny-sur-Orge",
    #        "Gare de Mennecy - Place de la Gare",
    #    ],
    # ),
]

for category, subcategory, names_to_remove in to_remove_tuple:
    geo_data_dict[f"{category}_{subcategory}_PER_POINT"].data = geo_data_dict[
        f"{category}_{subcategory}_PER_POINT"
    ].data.query("~name.isin(@names_to_remove)")

# %%
geo_data_dict.update(overall_h3_aggregation(geo_data_dict.copy(), base))
geo_data_dict.keys()

# %%
display(Markdown(f"### Overall scoring"))

# %%
geo_data_dict.update(
    overall_h3_scoring(
        geo_data_dict,
        base.copy(),
        weights=[
            1,  # Transit
            1,  # Destinations
            0.5,  # Lanes
            0.5,  # Bike parking
            1,  # Offices
            1,  # Population
        ],
    )
)

# %%
geo_data_dict["FINAL_SCORE_ALL_PER_HEX"].data.describe()

# %%
geo_data_dict["FINAL_SCORE_ALL_PER_HEX"].data.plot(
    cmap="magma",
    column="score",
    edgecolor="none",
    legend=True,
    figsize=(12, 8),
).axis("off")

# %%
total_pop = geo_data_dict["POPULATION_ALL_PER_HEX"].data.POPULATION.sum()
total_pop

# %%
display(Markdown(f"### Station selection"))

# %%
poi_hexes = get_poi_hexes(
    geo_data_dict,
    hex_resolution,
    skip_categories=["high_schools"],
)
poi_hexes = filter_poi_hexes(poi_hexes, base, market_vls)
poi_hexes

# %%
manual_hexes = []
manual_coords = [
    [49.05467600270687, 3.9301138674258604],  # Mairie Mardeuil
    [49.03985884140719, 3.9591037945343515],  # epernay résidentiel centre
]
manual_coords_hexes = get_manual_selection_hexes(manual_coords, hex_resolution)
manual_hexes.extend(manual_coords_hexes)
manual_hexes.extend(poi_hexes)
manual_hexes

# %%
hex_k_mapper = {9: 1, 10: 3}  # CAREFUL WITH THIS

# %%
import warnings

warnings.filterwarnings("ignore")

selected_sites = pd.DataFrame()
neighbors_list = []
for c in market_vls.LIBGEO.unique():
    print("\n====", c.upper(), "====")
    df_bs_sites, neighbors_list = clustered_sites(
        geo_data_dict["FINAL_SCORE_ALL_PER_HEX"].data[
            geo_data_dict["FINAL_SCORE_ALL_PER_HEX"].data["city_name"] == c
        ],
        target=market_vls[market_vls["LIBGEO"] == c]["stations"].values[0],
        score_col="score",
        hex_col=f"hex{hex_resolution}",
        neighbors_list=neighbors_list,
        rank_col="site_rank",
        ascending=False,
        k=hex_k_mapper[hex_resolution],
        hex_resolution=hex_resolution,
        max_distance=2000,  # RESET THIS AT 2000 OR 1000 FOR NEXT DEMAND ASSESSMENT
        manual_selection=manual_hexes,
    )
    selected_sites = selected_sites.append(df_bs_sites)
print("")
print(selected_sites.selected.value_counts(dropna=False))
selected_sites = selected_sites.query("selected == True")

# %%
display(Markdown(f"## Traffic predictions per stations"))

# %%
# method to calculate trips per station, from total expected trips gotten from model

selected_sites["score_ratio"] = selected_sites["score"] / selected_sites["score"].sum()

P_low_trips = market_vls["P_low_price_trips_Y"].sum()
O_low_trips = market_vls["O_low_price_trips_Y"].sum()
P_med_trips = market_vls["P_med_price_trips_Y"].sum()
O_med_trips = market_vls["O_med_price_trips_Y"].sum()
P_high_trips = market_vls["P_high_price_trips_Y"].sum()
O_high_trips = market_vls["O_high_price_trips_Y"].sum()

for scenario in ["P", "O"]:
    for price_range in ["low", "med", "high"]:
        selected_sites[f"station_{scenario}_{price_range}_trips"] = selected_sites[
            "score_ratio"
        ] * eval(f"{scenario}_{price_range}_trips")

# %%
expected_station_number = compute_stations_given_bikes(market_vls.rounded_bikes.sum())
extra_stations = expected_station_number - (len(selected_sites))

print(f"We have {expected_station_number} expected stations given our bike fleet size.")
print(f"We have selected {len(selected_sites)} stations for our demand assessment.")
print(
    f"Hence, {round((extra_stations /len(selected_sites)),2)*100}% of the proposed stations will have 20 bike spots."
)
print(f"{extra_stations} of the stations will have 20 bike spots.")
print(f"{len(selected_sites)-extra_stations} of the stations will have 10 bike spots.")

# %% tags=[]
selected_sites["Overall_Rank"] = selected_sites["score"].rank(ascending=0).astype(int)
selected_sites = selected_sites.assign(
    bike_spots=np.where(
        selected_sites.Overall_Rank <= extra_stations,
        20,
        10,
    )
)

# %%
selected_sites_20 = selected_sites[selected_sites["bike_spots"] == 20]
selected_sites_10 = selected_sites[selected_sites["bike_spots"] == 10]
print(len(selected_sites_20))
print(len(selected_sites_10))

# %%
display(Markdown(f"## Maps"))

# %%
add_Quicksand_font()

not_city = get_city_negative(city)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
_ = ax.axis("off")

# reset zoom level (due to not_city box)
xlim, ylim = get_map_limits(city)

# xlim = [823630.250588381, 838316.024679207]
# ylim = [6279944.366917977, 6294579.44074818]

ax.set_xlim(xlim)
ax.set_ylim(ylim)

geo_data_dict["FINAL_SCORE_ALL_PER_HEX"].data.query("score > score.quantile(.9)").plot(
    ax=ax,
    column="score",
    scheme="fisher_jenks_sampled",
    k=7,
    cmap="Blues",
    edgecolor="none",
    alpha=0.8,
)

print("plotted hexagons")

not_city.plot(
    ax=ax,
    color="white",
    edgecolor="none",
    alpha=0.4,
)
print("plotted city negative")

city.plot(ax=ax, color="none", edgecolor="black", linewidth=0.4, alpha=1)

print("plotted city boundary")

selected_sites_20.plot(
    ax=ax,
    color="red",
    edgecolor="none",
)

print("plotted selected sites for 20 spots")

selected_sites_10.plot(
    ax=ax,
    color="#e58213",
    edgecolor="none",
)

print("plotted selected sites for 10 spots")


# LEGEND
## legend elements
hexagon1 = Line2D(
    [0], [0], color="red", marker="h", linewidth=0, markersize=11, alpha=0.8
)
hexagon2 = Line2D(
    [0], [0], color="#e58213", marker="h", linewidth=0, markersize=11, alpha=0.8
)
line = Line2D([0], [0], color="black", linewidth=0.6, markersize=1, alpha=1)
cmap_gradient = get_gradient_cmap()

## labels

if len(selected_sites_20) > 0 and len(selected_sites_10) > 0:
    if country_code == "FR":
        labels = [
            "Potentiel emplacement station 20 spots",
            "Potentiel emplacement station 10 spots",
            "Score de demande latente",
            "Limite commune",
        ]
    else:
        labels = [
            "Bikeshare station 20 spots",
            "Bikeshare station 10 spots",
            "Bikeshare potential",
            "City limit",
        ]
    handles = [hexagon1, hexagon2, cmap_gradient, line]

elif len(selected_sites_20) > 0 and len(selected_sites_10) == 0:
    if country_code == "FR":
        labels = [
            "Potentiel emplacement station 20 spots",
            "Score de demande latente",
            "Limite commune",
        ]
    else:
        labels = ["Bikeshare station 20 spots", "Bikeshare potential", "City limit"]
    handles = [hexagon1, cmap_gradient, line]

else:
    if country_code == "FR":
        labels = [
            "Potentiel emplacement station 10 spots",
            "Score de demande latente",
            "Limite commune",
        ]
    else:
        labels = ["Bikeshare station 10 spots", "Bikeshare potential", "City limit"]
    handles = [hexagon2, cmap_gradient, line]

add_legend(ax, labels, handles, title=f"{place} VLS", location="best")
add_city_names(city, font_size=7)
add_north_arrow(ax)
add_scalebar(ax)

print("plotted map elements")


# basemap
cx.add_basemap(
    ax,
    crs=city.crs,
    # source =#cx.providers.Esri.WorldGrayCanvas ,#cx.providers.OpenStreetMap.Mapnik,#cx.providers.Stamen.TonerLite, #cx.providers.Stamen.Toner , #cx.providers.Stamen.Terrain
    source=cx.providers.Esri.WorldGrayCanvas,
    attribution=False,
    attribution_size=7,
)
print("plotted basemap")

plt.savefig(
    os.path.join(data_dir_out, f"{place.replace(' ','_')}_stations.png"),
    dpi=300,
    facecolor="w",
    bbox_inches="tight",
)

print("saved map")

# %%
map_color = fig

# %%
selected_sites_no_k_ring = selected_sites.copy()
selected_sites_no_k_ring.drop(columns="k_ring_3", inplace=True)

# %% tags=[]
m = selected_sites_no_k_ring.explore(
    "site_rank",
    height="100%",
    width="100%",
)
city.explore(m=m, style_kwds={"fill": False})

# %%
display(Markdown(f"## Exports"))

# %%
city_boundary = city.copy()  # .query("city_name.isin(@curr_city)").copy()
city_boundary.geometry = city_boundary.boundary

# %%
temp_dict = {
    "city": city_boundary,
    "stations": selected_sites[
        [
            "geometry",
            "site_rank",
            "city_name",
            f"hex{hex_resolution}",
            "station_P_low_trips",
            "station_P_med_trips",
            "station_P_high_trips",
            "station_O_low_trips",
            "station_O_med_trips",
            "station_O_high_trips",
        ]
    ],
}
temp_dict_20 = {
    "stations_20": selected_sites_20[
        [
            "geometry",
            "site_rank",
            "bike_spots",
            "city_name",
            f"hex{hex_resolution}",
            "station_P_low_trips",
            "station_P_med_trips",
            "station_P_high_trips",
            "station_O_low_trips",
            "station_O_med_trips",
            "station_O_high_trips",
        ]
    ],
}
temp_dict_10 = {
    "stations_10": selected_sites_10[
        [
            "geometry",
            "site_rank",
            "bike_spots",
            "city_name",
            f"hex{hex_resolution}",
            "station_P_low_trips",
            "station_P_med_trips",
            "station_P_high_trips",
            "station_O_low_trips",
            "station_O_med_trips",
            "station_O_high_trips",
        ]
    ],
}

# %%
save_gdf(temp_dict, data_dir_out, place)

# %%
save_gdf(temp_dict_20, data_dir_out, place)
save_gdf(temp_dict_10, data_dir_out, place)

# %% tags=[]
display(Markdown(f"## Deliverables"))

# %%
MAINTENANCE_BUFFER = 0.1

# %%
# | output: true
if country_code == "FR":
    display(Markdown("### Dimensionnement de la flotte\n\n"))
else:
    display(Markdown("### Fleet Sizing\n\n"))

# %%
# | output: true

print_vls_table(market_vls, country_code)

# %%
# | output: true
buffer_bikes = my_round(market_vls.rounded_bikes.sum() * MAINTENANCE_BUFFER, 10)

if country_code == "FR":
    display(
        Markdown(
            "__Taille de la flotte de vélos:__\n\n"
            f"- {market_vls.rounded_bikes.sum() + buffer_bikes} vélos totaux en libre service ({market_vls.rounded_bikes.sum()} vélos + {buffer_bikes} dédiés à la maintenance)\n"
            f"- {market_vld['bikes'].sum()} vélos en location longue durée \n"
            f"- {market_vls.stations.sum()} stations\n\n"
            # f"  - {len(selected_sites_20)} stations à 20 emplacements\n"
            f"  - {len(selected_sites_10)} stations à 10 emplacements"
        )
    )

else:
    display(
        Markdown(
            "__Overall bike share fleet:__\n\n"
            f"- {market_vls.rounded_bikes.sum() + buffer_bikes} total bikes for bike sharing ({market_vls.rounded_bikes.sum()} bikes + {buffer_bikes} maintenance buffer)\n"
            f"- {market_vld['bikes'].sum()} bikes for long term renting \n"
            f"- {market_vls.stations.sum()} stations\n\n"
            # f"  - {len(selected_sites_20)} stations w/ 20 spots\n"
            f"  - {len(selected_sites_10)} stations w/ 10 spots"
        )
    )

# %%
# | output: true
if country_code == "FR":
    display(Markdown("### Réseau de stations\n\n"))
else:
    display(Markdown("### Bikeshare network\n\n"))

# %%
# | output: true
map_color

# %%
# | output: true
display(
    Markdown(
        "Google My Maps [link](https://www.google.com/maps/d/edit?mid=1idYS0-BEidCr0i1noNCPqfO9GKJBQi4&usp=sharing)\n\n"
    )
)

# %%
# | output: true
if country_code == "FR":
    with pd.option_context("float_format", "{:.2f}".format):
        display(
            Markdown(
                "__Distance entre les stations__ \n\n"
                f"Interdistance moyenne : {round(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).mean())} mètres\n\n"
                f"Distance minimale : {(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).min()).astype(int)} mètres\n\n"
                f"Distance maximale : {(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).max()).astype(int)} mètres\n\n"
            )
        )
else:
    with pd.option_context("float_format", "{:.2f}".format):
        display(
            Markdown(
                "__Distance between stations__ \n\n"
                f"Average distance : {round(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).mean())} meters\n\n"
                f"Minimum distance : {(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).min()).astype(int)} meters\n\n"
                f"Maximum distance : {(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).max()).astype(int)} meters\n\n"
            )
        )

# %%
# | output: true

if country_code == "FR":
    display(Markdown("__Approximation de la couverture de population__\n\n"))
else:
    display(Markdown("__Approximate population coverage__\n\n"))

for buffer in [400, 1000]:

    covered_pop = get_pop_coverage(
        selected_sites,
        geo_data_dict["POPULATION_ALL_PER_POINT"].data,
        local_crs,
        buffer=buffer,
    )
    if country_code == "FR":
        display(
            Markdown(
                f"Population à moins de {buffer}m d'une station : {covered_pop} soit environ {round(covered_pop/total_pop*100)}% de la population"
            )
        )
    else:
        display(
            Markdown(
                f"People living less than {buffer}m from a station : {covered_pop} being around {round(covered_pop/total_pop*100)}% of total population"
            )
        )

# %%
# Google my maps legends
if country_code == "FR":
    output = (
        f"{place} - Proposition Réseau VLS\n\n"
        "Emplacement approx. stations VLS\n\n"
        "Limites communes\n\n"
    )
else:
    output = (
        f"{place} - Bike Share Network Proposal\n\n"
        "Approx. bike station location\n\n"
        "City boundary"
    )

Markdown(output)

# %%
# | output: true
if country_code == "FR":
    display(Markdown("### Prédictions d'indicateurs de performance\n\n\n"))
else:
    display(Markdown("### Performance prediction indicators\n\n\n"))

# %%
predictive_db = get_base_predictions_db(market_vls)
with pd.option_context("display.max_rows", 10, "display.max_columns", None):
    display(predictive_db)

# %%
kpi_db = get_kpi_db_market(predictive_db)
kpi_db

# %%
PRICE_PER_BIKE = 1800

kpi_db = get_bike_rotation(kpi_db)
kpi_db = get_maintenance_costs(kpi_db)
kpi_db.head(1)

# %%
# | output: true

print_predictions_header(
    kpi_db,
    country_code,
)

# %% [raw] tags=[]
# ::: {.panel-tabset}
#
# ###  Prix bas

# %%
# | output: true
print_kpi_predictions_panel(kpi_db, country_code, ["low"])

# %% [raw] tags=[]
# ###  Prix moyen

# %%
# | output: true
print_kpi_predictions_panel(kpi_db, country_code, ["med"])

# %% [raw]
# ###  Prix haut

# %%
# | output: true
print_kpi_predictions_panel(kpi_db, country_code, ["high"])

# %% [raw]
# :::

# %%
# | output: true
if country_code == "FR":
    display(
        Markdown("### Tableau de bord des prédictions d'indicateurs de performance\n\n")
    )
else:
    display(Markdown("### KPIs prediction dashboard\n\n"))

# %%
# | output: true
# | column: screen

get_plotly_KPIs(kpi_db, country_code, dashboard_height_margin=400)

# %%
# app = JupyterDash(__name__)
# app.layout = html.Div(
#     children=[
#         dcc.Graph(
#             id="kpi_dashboard",
#             figure=get_plotly_KPIs(kpi_db),
#         ),
#         dcc.Checklist(
#             id="town_checklist",
#             options=kpi_db["LIBGEO"],
#             value=kpi_db["LIBGEO"],
#         ),
#     ],
#     style={"height": "90%", "width": "40%"},
# )


# @app.callback(Output("kpi_dashboard", "figure"), Input("town_checklist", "value"))
# def update_fig(outvalues):
#     kpi_db_filtered = kpi_db.query("LIBGEO in @outvalues")
#     fig = get_plotly_KPIs(kpi_db_filtered)

#     return fig

# app.run_server(mode="inline", debug=True, port=8072)
