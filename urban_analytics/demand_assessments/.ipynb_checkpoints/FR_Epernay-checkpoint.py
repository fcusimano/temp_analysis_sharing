# ---
# format: 
#   pdf:
#     echo: false
#     output: false
#     documentclass: report
#     fontfamily: opensans
#   html:
#     echo: false
#     output: false
#     page-layout: full
# jupyter: python3
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
def load_city_and_fleet_info():
    french_vls_fleet_path = "../data/in/final_VLS_fleet.feather"
    return pd.read_feather(french_vls_fleet_path)


# %%
def get_city_selection_df_vincent(
    target_city_name: List[str], city_subset: List[str] = ["*"]
) -> pd.DataFrame:
    """
    Returns dataframe with selected cities' info.
    If no `city_subset` defined, will return all cities in the area_name (i.e. 'intercommunalité').
    Will raise alert if not all `city_subset` cities are found (in whic case you probably need to also add them to `target_city_name`)
    # TODO: Ignore differences in accents, caps and dashes.

    Output df columns: `city_name`, `area_name`, `dept_id`, `dept_name`, `region_id`, `region_name`

    Arguments:
        target_city_name: main city(ies) to study, they should ideally not have a homonym in France
        city_subset: add cities here if you do not want to keep all cities within the returned area(s)

    Examples:
        >>> from urban_analytics.fleet_sizing import get_city_selection_df
        >>>
        >>> output = get_city_selection_df(
        ...     target_city_name=["Saclay"], city_subset=["Massy", "Saclay"]
        ... )
        All cities found (2)
        1 area(s) (CA Communauté Paris-Saclay)
        >>>
        >>> output
              city_name                   area_name dept_name dept_id    region_name  region_id
        34441     Massy  CA Communauté Paris-Saclay   ESSONNE      91  ILE DE FRANCE         11
        34481    Saclay  CA Communauté Paris-Saclay   ESSONNE      91  ILE DE FRANCE         11
    """
    french_vls_file = load_city_and_fleet_info()
    agglo_list = french_vls_file.query(
        "LIBGEO.isin(@target_city_name)"
    ).LIBEPCI.tolist()
    agglo_df = french_vls_file.query("LIBEPCI.isin(@agglo_list)")[
        [
            "LIBGEO",
            "LIBEPCI",
            "DEP",
            "dep_name",
            "population",
            "area_km2",
            "REG",
            "region_name",
            "base_bikes",
            "clients_low_est",
            "clients_high_est",
            "P_low_price_trips_Y",
            "P_med_price_trips_Y",
            "P_high_price_trips_Y",
            "O_low_price_trips_Y",
            "O_med_price_trips_Y",
            "O_high_price_trips_Y",
            "P_low_km_Y",
            "P_med_km_Y",
            "P_high_km_Y",
            "O_low_km_Y",
            "O_med_km_Y",
            "O_high_km_Y",
            "P_low_kg_CO2_saved",
            "P_med_kg_CO2_saved",
            "P_high_kg_CO2_saved",
            "O_low_kg_CO2_saved",
            "O_med_kg_CO2_saved",
            "O_high_kg_CO2_saved",
            "surplus_workers",
            "final_pop_coeff",
            "final_cycl_coeff",
            "fub_score",
            "cost_km",
            "bike_replacement_Y",
            "O_trips_from_tourism",
            "P_trips_from_tourism",
        ]
    ]
    if "*" not in city_subset:
        cities_df = agglo_df.query("LIBGEO.isin(@city_subset)")
        if len(cities_df) == len(city_subset):
            print(f"All cities found ({len(cities_df)})")
        else:
            print(
                f"Missing {len(city_subset)-len(cities_df)} city(ies): {', '.join((set(cities_df.LIBGEO) ^ set(city_subset)))}"
            )
    else:
        cities_df = agglo_df.copy()
        print(f"{len(cities_df)} cities found")
    print(
        f"{len(cities_df.LIBEPCI.unique())} area(s) ({','.join(cities_df.LIBEPCI.unique())})"
    )
    return cities_df


# %%
place = "Épernay"
country_code = "FR"

local_crs = LocalCRS[country_code].value
data_dir_in, data_dir_out = set_up_directories(country_code, place)

# %%
# | output: true
if country_code == "FR":
    display(Markdown(f"# {place} Etude de territoire\n\n"))
else:
    display(Markdown(f"# {place} Demand Assessment\n\n"))

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
target_cities_df = get_city_selection_df_vincent(
    target_city_name=[
        "Épernay",
    ],
    city_subset=["Épernay", "Magenta", "Moussy", "Pierry", "Vinay", "Mardeuil"],
)


# %%
def get_vls_fleet_vincent(fr_fleet_df: pd.DataFrame, batches: int = 10) -> pd.DataFrame:
    """
    Compute bikesharing fleet for target cities based on precomputed stats (fr_vls).

    Arguments:
        cities_df: city dataframe obtained from [`get_city_selection_df()`][urban_analytics.fleet_sizing.get_city_selection_df]
        batches: bike fleet(s) will be a multiple of this value

    Examples:
        >>> # get_vls_fleet(cities_df)
        #          city    pop  ... bikes  stations
        # 1       Massy  40309  ...    80        14
        # 2      Saclay   3054  ...     0         0
        ```
    """
    fr_fleet_df["rounded_bikes"] = 10 * round(fr_fleet_df["base_bikes"] / 10).astype(
        int
    )
    fr_fleet_df["stations"] = round(
        fr_fleet_df["rounded_bikes"].apply(lambda x: x / 10 * 1.8)
    ).astype(int)
    return fr_fleet_df


# %%
market_vls = get_vls_fleet_vincent(target_cities_df, batches=10)
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
def get_specific_place_name_vincent(target_cities_df: pd.DataFrame) -> List[str]:
    """
    Returns string of city, department and region name from French city dataframe to
    avoid finding duplicates when querying OSM.

    Arguments:
        target_cities_df: dataframe with city informations

    Examples:
        >>> import pandas as pd
        >>> from urban_analytics.utils import get_specific_place_name
        >>> input = pd.DataFrame({
        ...          "city_name": ["Massy"],
        ...          "area_name": ["CA Communauté Paris-Saclay"],
        ...          "dept_name": ["ESSONNE"],
        ...          "dept_id": ["91"],
        ...          "region_name": ["ILE DE FRANCE"],
        ...          "region_id": [11],
        ...      })
        >>> get_specific_place_name(input)
        ['Massy,ESSONNE,ILE DE FRANCE']
    """
    return (
        target_cities_df.loc[:, ["LIBGEO", "dep_name", "region_name"]]
        .apply(lambda x: ",".join(x), axis=1)
        .unique()
        .tolist()
    )


# %%
specific_place_name = get_specific_place_name_vincent(target_cities_df)
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
def filter_poi_hexes_vincent(
    poi_hexes: List[str],
    base: gpd.GeoDataFrame,
    market_vls: pd.DataFrame,
) -> List[str]:
    """
    Remove point of interest hexes in cities where stations are not needed.

    Arguments:
        poi_hexes: list of point of interest hexes (obtained with [`get_poi_hexes`][urban_analytics.station_selection.get_poi_hexes])
        base: hexagon base for the study area (obtained with [`get_h3_fill`][urban_analytics.h3_utils.get_h3_fill]
        market_vls: bike share fleet sizing dataframe with final number of stations per city

    Returns:
        List point of interest hexes for cities where stations are needed
    """
    hex_resolution = base.h3_res[0]
    hex_col = f"hex{hex_resolution}"
    return list(
        set(poi_hexes)
        & set(
            base.loc[
                base["city_name"].isin(market_vls.query("stations > 0").LIBGEO.unique())
            ][hex_col].unique()
        )
    )


# %%
poi_hexes = get_poi_hexes(
    geo_data_dict,
    hex_resolution,
    skip_categories=["high_schools"],
)
poi_hexes = filter_poi_hexes_vincent(poi_hexes, base, market_vls)

# %%
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
def save_gdf(gdf_dict, folder: str, city_name: str):
    """
    Saves geodaframe(s) to kml format.

    Arguments:
        gdf_dict: dictionary of geodataframe(s) variable names as keys and geodataframe(s) as values
        folder: export directory

    Examples:
        >>> # from urban_analytics.geography import save_gdf
        >>> # save_gdf(["city_gdf","stations_gdf"], 'kml_exports')

    """
    fiona.supported_drivers["KML"] = "rw"

    for key, value in gdf_dict.items():
        path_to_file = os.path.join(folder, f"{city_name}_{key}.kml")
        if os.path.exists(path_to_file):
            os.remove(path_to_file)
        value.to_file(path_to_file, driver="KML")
        print(path_to_file, "created")


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

html_market = market_vls[["LIBGEO", "population", "rounded_bikes", "stations"]]


if country_code == "FR":
    html_market.rename(
        columns={
            "LIBGEO": "Communes",
            "population": "Population",
            "rounded_bikes": "Vélos",
            "stations": "Stations",
        },
        inplace=True,
    )
    html_market.set_index("Communes", inplace=True)
else:
    html_market.rename(
        columns={
            "LIBGEO": "Towns",
            "population": "Population",
            "rounded_bikes": "Bikes",
            "stations": "Stations",
        },
        inplace=True,
    )
    html_market.set_index("Towns", inplace=True)
html_market.loc["Total"] = html_market.sum(numeric_only=True, axis=0)
html_market

# %%
# | output: true
buffer_bikes = my_round(market_vls.rounded_bikes.sum() * MAINTENANCE_BUFFER, 10)

if country_code == "FR":
    display(
        Markdown(
            "__Taille de la flotte de vélos:__\n\n"
            f"- {market_vls.rounded_bikes.sum() + buffer_bikes} vélos totaux ({market_vls.rounded_bikes.sum()} vélos + {buffer_bikes} dédiés à la maintenance)\n"
            f"- {market_vls.stations.sum()} stations\n\n"
            # f"  - {len(selected_sites_20)} stations à 20 emplacements\n"
            f"  - {len(selected_sites_10)} stations à 10 emplacements"
        )
    )

else:
    display(
        Markdown(
            "__Overall bike share fleet:__\n\n"
            f"- {market_vls.rounded_bikes.sum() + buffer_bikes} total bikes ({market_vls.rounded_bikes.sum()} bikes + {buffer_bikes} maintenance buffer)\n"
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
    display(Markdown("__Distance entre les stations__ \n\n"))
else:
    display(Markdown("__Distance between stations__ \n\n"))

# %%
# | output: true
if country_code == "FR":
    with pd.option_context("float_format", "{:.2f}".format):
        display(
            Markdown(
                f"Distance moyenne entre les stations les plus proches : {round(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).mean())} mètres\n\n"
                f"Distance minimale entre les stations les plus proches : {(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).min()).astype(int)} mètres\n\n"
                f"Distance maximale entre les stations les plus proches : {(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).max()).astype(int)} mètres\n\n"
            )
        )
else:
    with pd.option_context("float_format", "{:.2f}".format):
        display(
            Markdown(
                f"Average distance between closest stations : {round(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).mean())} meters\n\n"
                f"Minimum distance between closest stations : {(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).min()).astype(int)} meters\n\n"
                f"Maximum distance between closest stations : {(weighted_dist_to_next_site(selected_sites, weights=[1, 0]).max()).astype(int)} meters\n\n"
            )
        )

# %%
# | output: true
if country_code == "FR":
    display(Markdown("__Approximation de la couverture de population__\n\n"))
else:
    display(Markdown("__Approximate population coverage__\n\n"))

# %%
# | output: true

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
def get_base_predictions_db(base_pop_file):
    return base_pop_file[
        [
            "LIBGEO",
            "CODGEO",
            "LIBEPCI",
            "population",
            "surplus_workers",
            "DEP",
            "rounded_bikes",
            "base_bikes",
            "final_pop_coeff",
            "final_cycl_coeff",
            "fub_score",
            "cost_km",
            "bike_replacement_Y",
            "clients_low_est",
            "clients_high_est",
            "O_trips_from_tourism",
            "O_low_price_trips_Y",
            "O_med_price_trips_Y",
            "O_high_price_trips_Y",
            "O_low_km_Y",
            "O_med_km_Y",
            "O_high_km_Y",
            "O_low_kg_CO2_saved",
            "O_med_kg_CO2_saved",
            "O_high_kg_CO2_saved",
            "P_trips_from_tourism",
            "P_low_price_trips_Y",
            "P_med_price_trips_Y",
            "P_high_price_trips_Y",
            "P_low_km_Y",
            "P_med_km_Y",
            "P_high_km_Y",
            "P_low_kg_CO2_saved",
            "P_med_kg_CO2_saved",
            "P_high_kg_CO2_saved",
        ]
    ].sort_values("population", ascending=False)


# %%
predictive_db = get_base_predictions_db(market_vls)
with pd.option_context("display.max_rows", 10, "display.max_columns", None):
    display(predictive_db)


# %%
def get_kpi_db_market(prediction_db, proposed_bikes=["*"]):
    base_bikes = round(prediction_db["rounded_bikes"].values.sum())
    if proposed_bikes == ["*"]:
        proposed_bikes = base_bikes
    bike_fleet_delta = (
        (base_bikes - proposed_bikes) / 2 + proposed_bikes
    ) / base_bikes  # used to adjust KPIs based on proposed size of bike fleet
    kpi_db = prediction_db.copy()
    kpi_db[kpi_db.columns[-22:]] = round(
        kpi_db[kpi_db.columns[-22:]].multiply(bike_fleet_delta, axis="index")
    ).astype(int)
    kpi_db["proposed_bikes"] = (
        kpi_db["rounded_bikes"] / kpi_db["rounded_bikes"].sum() * proposed_bikes
    ).astype(int)
    kpi_db.insert(6, "proposed_bikes", kpi_db.pop("proposed_bikes"))
    return kpi_db.reset_index(drop=True)


# %%
kpi_db = get_kpi_db_market(predictive_db)
kpi_db


# %%
def get_bike_rotation(kpi_df):
    for scenario in [
        ("P"),
        ("O"),
    ]:
        for price_type in [
            ("low"),
            ("med"),
            ("high"),
        ]:
            kpi_df[f"{scenario}_{price_type}_day_rotation"] = round(
                kpi_df[f"{scenario}_{price_type}_price_trips_Y"]
                / kpi_df["proposed_bikes"]
                / 365,
                2,
            )
    return kpi_df


# %%
def get_maintenance_costs(kpi_df):
    for scenario in [
        ("P"),
        ("O"),
    ]:
        for price_type in [
            ("low"),
            ("med"),
            ("high"),
        ]:
            kpi_df[f"{scenario}_{price_type}_cost_Y"] = round(
                # maintenance of parts
                (kpi_df[f"{scenario}_{price_type}_km_Y"] * kpi_df["cost_km"])
                +
                # bike replacements
                (
                    kpi_df["proposed_bikes"]
                    * (kpi_df["bike_replacement_Y"])
                    * PRICE_PER_BIKE
                )
            ).astype(int)
    return kpi_df


# %%
PRICE_PER_BIKE = 1800

kpi_db = get_bike_rotation(kpi_db)
kpi_db = get_maintenance_costs(kpi_db)
kpi_db.head(1)


# %%
def print_kpi_predictions(kpi_df):
    text = []
    city_name = kpi_df["LIBGEO"].values[0]
    proposed_bikes = round(kpi_df["proposed_bikes"].values.sum())
    text += [
        f"\n__Pessimistic and optimistic scenarios for {city_name} with {proposed_bikes} proposed bikes__\n",
        f"Expected local users per year: {round(kpi_df['clients_low_est'].values.sum()):,} to {round(kpi_df['clients_high_est'].values.sum()):,}",
    ]
    for price_type, price_type_var, tourist_price_coeff in [
        ("Low", "low", LOW_PRICE_IMPACT_TOURISM_CONVERSION),
        ("Medium", "med", MEDIUM_PRICE_IMPACT_TOURISM_CONVERSION),
        ("High", "high", HIGH_PRICE_IMPACT_TOURISM_CONVERSION),
    ]:
        scenario_type = ["P", "O"]

        text += [f"\n{price_type} pricing estimation\n"]
        text += [
            f"- Expected trips per year: {round(kpi_df[f'{scenario_type[0]}_{price_type_var}_price_trips_Y'].values.sum()):,} ({int(kpi_df['P_trips_from_tourism'].values.sum()*tourist_price_coeff):,} from tourism) to {round(kpi_df[f'{scenario_type[1]}_{price_type_var}_price_trips_Y'].values.sum()):,} ({int(kpi_df['O_trips_from_tourism'].values.sum()*tourist_price_coeff):,} from tourism)",
            f"- Expected bike daily rotation: {round(((kpi_df[f'{scenario_type[0]}_{price_type_var}_price_trips_Y'].values.sum())/365/(kpi_df['proposed_bikes'].values.sum())),2)} to {round(((kpi_df[f'{scenario_type[1]}_{price_type_var}_price_trips_Y'].values.sum())/365/(kpi_df['proposed_bikes'].values.sum())),2)}",
            f"- Expected km per year: {round(kpi_df[f'{scenario_type[0]}_{price_type_var}_km_Y'].values.sum()):,} to {round(kpi_df[f'{scenario_type[1]}_{price_type_var}_km_Y'].values.sum()):,}",
            f"- Expected maintenance costs per year: {round(kpi_df[f'{scenario_type[0]}_{price_type_var}_cost_Y'].values.sum()):,}€ to {round(kpi_df[f'{scenario_type[1]}_{price_type_var}_cost_Y'].values.sum()):,}€",
            f"- Expected saved CO2 kg per year: {round(kpi_df[f'{scenario_type[0]}_{price_type_var}_kg_CO2_saved'].values.sum()):,} to {round(kpi_df[f'{scenario_type[1]}_{price_type_var}_kg_CO2_saved'].values.sum()):,}",
        ]
    return Markdown("\n".join(text))


# %%
def print_kpi_predictions_fr(kpi_df):
    text = []
    city_name = kpi_df["LIBGEO"].values[0]
    proposed_bikes = round(kpi_df["proposed_bikes"].values.sum())
    text += [
        f"\n__Scénarios pessimistes et optimistes pour {city_name} avec une proposition de {proposed_bikes} vélos__\n",
        f"Utilisateurs attendus par an: {round(kpi_df['clients_low_est'].values.sum()):,} à {round(kpi_df['clients_high_est'].values.sum()):,}",
    ]
    for price_type, price_type_var, tourist_price_coeff in [
        ("bas", "low", LOW_PRICE_IMPACT_TOURISM_CONVERSION),
        ("moyen", "med", MEDIUM_PRICE_IMPACT_TOURISM_CONVERSION),
        ("élevé", "high", HIGH_PRICE_IMPACT_TOURISM_CONVERSION),
    ]:
        scenario_type = ["P", "O"]

        text += [f"\n Estimation de politique de prix {price_type}\n"]
        text += [
            f"- Trajets attendus par an: {round(kpi_df[f'{scenario_type[0]}_{price_type_var}_price_trips_Y'].values.sum()):,} ({int(kpi_df['P_trips_from_tourism'].values.sum()*tourist_price_coeff):,} liés au tourisme) à {round(kpi_df[f'{scenario_type[1]}_{price_type_var}_price_trips_Y'].values.sum()):,} ({int(kpi_df['O_trips_from_tourism'].values.sum()*tourist_price_coeff):,} liés au tourisme)",
            f"- Rotation journalière par vélo attendue: {round(((kpi_df[f'{scenario_type[0]}_{price_type_var}_price_trips_Y'].values.sum())/365/(kpi_df['proposed_bikes'].values.sum())),2)} à {round(((kpi_df[f'{scenario_type[1]}_{price_type_var}_price_trips_Y'].values.sum())/365/(kpi_df['proposed_bikes'].values.sum())),2)}",
            f"- Kilomètres attendus par an: {round(kpi_df[f'{scenario_type[0]}_{price_type_var}_km_Y'].values.sum()):,} à {round(kpi_df[f'{scenario_type[1]}_{price_type_var}_km_Y'].values.sum()):,}",
            f"- Couts de maintenance attendus par an: {round(kpi_df[f'{scenario_type[0]}_{price_type_var}_cost_Y'].values.sum()):,}€ à {round(kpi_df[f'{scenario_type[1]}_{price_type_var}_cost_Y'].values.sum()):,}€",
            f"- Kilos de CO2 évités attendus par an: {round(kpi_df[f'{scenario_type[0]}_{price_type_var}_kg_CO2_saved'].values.sum()):,} à {round(kpi_df[f'{scenario_type[1]}_{price_type_var}_kg_CO2_saved'].values.sum()):,}",
        ]
    return Markdown("\n".join(text))


# %%
# | output: true

LOW_PRICE_IMPACT_TOURISM_CONVERSION = 1.4
MEDIUM_PRICE_IMPACT_TOURISM_CONVERSION = 1
HIGH_PRICE_IMPACT_TOURISM_CONVERSION = 0.75

print_kpi_predictions_fr(kpi_db)

# %%
# | output: true
if country_code == "FR":
    display(
        Markdown("### Tableau de bord des prédictions d'indicateurs de performance\n\n")
    )
else:
    display(Markdown("### KPIs prediction dashboard\n\n"))


# %%
def get_plotly_KPIs(kpi_db, wish_to_save=False):
    bike_fleet_delta = (
        (kpi_db["base_bikes"].values.sum() - kpi_db["proposed_bikes"].values.sum()) / 2
        + kpi_db["proposed_bikes"].values.sum()
    ) / kpi_db["base_bikes"].values.sum()

    if country_code == "FR":
        title_text = f"Indicateurs attendus pour les communes choisies avec {round(kpi_db['proposed_bikes'].sum())} vélos (multiplicateur d'impact de {round((bike_fleet_delta),2)} comparé à une flotte optimale de {round(kpi_db['base_bikes'].values.sum())} vélos)"
        subplot_titles = (
            "Utilisateurs attendus par an",
            "Trajets attendus par an",
            "Kilomètres attendus par an",
            "CO2 évité (kg) attendu par an",
            "Rotation par vélo attendue par jour",
            "Couts de maintenance (€) attendus par an",
        )
    else:
        title_text = f"Expected KPIs for selected cities with {round(kpi_db['proposed_bikes'].sum())} bikes (impact multiplier of {round((bike_fleet_delta),2)} compared to optimal fleet of {round(kpi_db['base_bikes'].values.sum())} bikes)"
        subplot_titles = (
            "Expected users per year",
            "Expected trips per year",
            "Expected kilometers per year",
            "Expected saved kg CO2 per year",
            "Expected bike rotation per day",
            "Expected maintenance costs (€) per year",
        )

    fig = make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.15,
        subplot_titles=subplot_titles,
    )

    for scenario_type, marker_color, legend_name in [
        ("low", "lightblue", "Pessimistic"),
        ("high", "darkblue", "Optimistic"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=kpi_db[f"clients_{scenario_type}_est"],
                y=kpi_db["LIBGEO"],
                text=[f"{legend_name}"] * len(kpi_db),
                mode="markers",
                marker_color=marker_color,
                marker_size=10,
                name=f"{legend_name} users",
                hovertemplate="""Town: %{y} <br> %{text} users: %{x} <br><extra></extra>""",
                legendgroup="1",
            ),
            row=1,
            col=1,
        )

    for index, rows in kpi_db.iterrows():
        fig.add_shape(
            type="line",
            x0=kpi_db["clients_low_est"][index],
            y0=kpi_db["LIBGEO"][index],
            x1=kpi_db["clients_high_est"][index],
            y1=kpi_db["LIBGEO"][index],
            line=dict(color="blue", width=2),
            row=1,
            col=1,
        )

    scenario_type = ["P", "O"]

    for price_type, price_type_name, line_color, legendgroup in [
        ("low", "low price", "green", 2),
        ("med", "medium price", "orange", 3),
        ("high", "high price", "red", 4),
    ]:
        for KPI, KPI_legendname, plot_row, plot_column, show_legend in [
            ("price_trips_Y", "trips", 1, 2, True),
            ("km_Y", "kilometers", 2, 1, False),
            ("kg_CO2_saved", "saved CO2", 2, 2, False),
            ("day_rotation", "daily rotation", 3, 1, False),
            ("cost_Y", "maintenance costs", 3, 2, False),
        ]:
            for index, rows in kpi_db.iterrows():
                fig.add_shape(
                    type="line",
                    x0=kpi_db[f"P_{price_type}_{KPI}"][index],
                    y0=kpi_db["LIBGEO"][index],
                    x1=kpi_db[f"O_{price_type}_{KPI}"][index],
                    y1=kpi_db["LIBGEO"][index],
                    line=dict(color=line_color, width=2),
                    row=plot_row,
                    col=plot_column,
                )
            for scenario, opacity in [
                ("Pessimistic", 0.5),
                ("Optimistic", 1),
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=kpi_db[f"{scenario[0]}_{price_type}_{KPI}"],
                        y=kpi_db["LIBGEO"],
                        text=[f"{scenario} {price_type_name} {KPI_legendname}"]
                        * len(kpi_db),
                        mode="markers",
                        marker_color=line_color,
                        opacity=opacity,
                        marker_size=10,
                        name=f"{scenario} {price_type_name} KPIs",
                        hovertemplate="""Town: %{y} <br> %{text}: %{x} <br><extra></extra>""",
                        legendgroup=legendgroup,
                        showlegend=show_legend,
                    ),
                    row=plot_row,
                    col=plot_column,
                )

    fig.update_layout(
        {
            "plot_bgcolor": "rgba(183,168,185, 0.15)",
        },
        height=80 * len(kpi_db) + 500,
        width=1600,
        legend_tracegroupgap=15,
        title_text=title_text,
        title_font_size=22,
    )
    fig.update_xaxes(
        showline=False,
        linewidth=2,
        linecolor="black",
        gridcolor="lightgrey",
        gridwidth=0.5,
    )
    fig.update_yaxes(
        showline=False,
        linewidth=2,
        linecolor="black",
        gridcolor="lightgrey",
        gridwidth=0.5,
        categoryorder="total ascending",
    )

    fig.update_traces(
        marker=dict(
            size=(
                (
                    MinMaxScaler(feature_range=(0.2, 0.5)).fit_transform(
                        kpi_db[["population"]]
                    )
                    * 30
                )
            ),
            symbol="diamond",
            line=dict(width=0.2, color="black"),
        ),
    )
    if wish_to_save == True:
        fig.write_html(
            f"datasets/output/dashboard_{kpi_db['LIBGEO'][0]}_{round(kpi_db['proposed_bikes'].values.sum())}.html"
        )
    return fig


# %%
get_plotly_KPIs(kpi_db, False)

# %%
# | output: true
# | column: screen

app = JupyterDash(__name__)

app.layout = html.Div(
    children=[
        dcc.Graph(
            id="kpi_dashboard",
            figure=get_plotly_KPIs(kpi_db),
        ),
        dcc.Checklist(
            id="town_checklist",
            options=kpi_db["LIBGEO"],
            value=kpi_db["LIBGEO"],
        ),
    ],
    style={"height": "90%", "width": "40%"},
)


@app.callback(Output("kpi_dashboard", "figure"), Input("town_checklist", "value"))
def update_fig(outvalues):
    kpi_db_filtered = kpi_db.query("LIBGEO in @outvalues")
    fig = get_plotly_KPIs(kpi_db_filtered)

    return fig


app.run_server(mode="inline", debug=True, port=8072)

# %%
