# ---
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

# %% [markdown] tags=[]
# # Auxerre AO
# ### Setup
#

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

from matplotlib.lines import Line2D
import contextily as cx

# %load_ext lab_black

# %%
def load_city_and_fleet_info():
    french_vls_fleet_path = "/home/vincentaguerrechariol/Documents/repos/data-notebook-sharing/urban_analytics/others/open_territories/datasets/final_VLS_fleet.feather"
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
place = "Auxerre"
country_code = "FR"

local_crs = LocalCRS[country_code].value
data_dir_in, data_dir_out = set_up_directories(country_code, place)

# %% [markdown]
# ## City selection and fleet sizing
# # VLS market

# %%
target_cities_df = get_city_selection_df_vincent(
    target_city_name=[
        "Auxerre",
    ],
    # city_subset=[],
)


# %% [markdown] tags=[]
# ## Fleet sizing
# ### VLS market

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
market_vls.sort_values("rounded_bikes", ascending=False)

# %%
# manual parameters for 220 bikes and 42 stations
market_vls["rounded_bikes"] = 0
market_vls["rounded_bikes"][2] = 220
market_vls["stations"] = 1
market_vls["stations"][2] = 14
market_vls = market_vls.sort_values("rounded_bikes", ascending=False)
market_vls


# %% [markdown]
# ### __Long-term rental scheme (VLD)__

# %% [raw]
# market_vld = get_vld_fleet(target_cities_df, batches=5)
# print("Recommended long bike rental fleet size by city:")
# print(
#     f"{len(market_vld)} cities and {int(market_vld.bikes_lower.sum())}-{int(market_vld.bikes_upper.sum())} bikes"
# )
# print_fleet_df(
#     market_vld[
#         [
#             "city",
#             "bikes",
#         ]
#     ]
# )

# %% [markdown]
# ## Geography

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

# %% [markdown]
# ## H3

# %%
hex_resolution = 10

# Generate H3 hexagons
base = get_h3_fill(city, hex_resolution, local_crs)

# Plot
ax = base.plot(figsize=(8, 8))
city.plot(ax=ax, color="None").axis("off")

# %% [markdown]
# ## External data

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
    #        "IFSI de Perray-Vaucluse - Université Paris-Saclay",
    #        "IFSI de Perray-Vaucluse",
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

# %% [markdown]
# ## Overall scoring

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


# %% [markdown] tags=[]
# ## Station selection

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
to_remove_poi = [
    "8a1fb388489ffff",
    "8a1fb3884007fff",
    "8a1fb38a3cf7fff",
    "8a1fb38ae527fff",
    "8a1fb38ae2effff",
    "8a1fb38acb27fff",
    "8a1fb389c337fff",
    "8a1fb3889a9ffff",
    "8a1fb3d6805ffff",
    "8a1fb388060ffff",
    "8a1fb3880817fff",
]

# %%
poi_hexes = [item for item in poi_hexes if item not in to_remove_poi]
poi_hexes

# %%
manual_hexes = []
manual_coords = [
    [47.79718391003021, 3.574748843973096],  # quai place de la république
    [47.78824710509519, 3.585343000086406],  # stade
    [47.797582952927094, 3.557766609781977],  # aldi ouest ville
    [47.805059352603884, 3.5549368432516517],  # hopital + centre commercial
    [47.80657908580908, 3.5652209887542425],  # auxerre nord
    [47.78908514858516, 3.5613700609568664],  # auxerre sud
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
        # max_distance=2000,  # RESET THIS AT 2000 OR 1000 FOR NEXT DEMAND ASSESSMENT
        manual_selection=manual_hexes,
    )
    selected_sites = selected_sites.append(df_bs_sites)
print("")
print(selected_sites.selected.value_counts(dropna=False))
selected_sites = selected_sites.query("selected == True")

# %%
# assert_station_selection(selected_sites, market_vls)

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
        selected_sites.Overall_Rank + 6 <= extra_stations,
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
selected_sites_final = selected_sites.copy()
selected_sites_final = selected_sites_final.sort_values("Overall_Rank", ascending=False)
selected_sites_final

# %%
gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
selected_stations = gpd.read_file(
    "/home/vincentaguerrechariol/Documents/repos/data-notebook-sharing/urban_analytics/data/in/FR/Auxerre/Emplacement des stations.kml",
    driver="KML",
)
selected_stations.set_crs(WGS84_CRS, inplace=True)
selected_stations.to_crs(local_crs, inplace=True)
selected_stations

# %% [markdown]
# ## Maps

# %%
add_Quicksand_font()

not_city = get_city_negative(city)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
_ = ax.axis("off")

# reset zoom level (due to not_city box)
xlim, ylim = get_map_limits(city)
# xlim = [739869.2674018644, 746478.3510588681]
# ylim = [6741688.72190895, 6748216.922225068]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

geo_data_dict["FINAL_SCORE_ALL_PER_HEX"].data.query(
    "score > score.quantile(.995)"
).plot(
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

# selected_sites_10.plot(
#    ax=ax,
#   color="red",
#    edgecolor="none",
# )

# print("plotted selected sites for 10 spots")

selected_stations.plot(
    ax=ax, color="#FF0062", edgecolor="white", markersize=35, linewidth=1
)

print("plotted selected stations")


# LEGEND
## legend elements
hexagon1 = hexagon1 = Line2D(
    [0],
    [0],
    color="#FF0062",
    marker="o",
    markersize=8,
    markeredgecolor="white",
    linewidth=0,
    markeredgewidth=0.3,
)
hexagon2 = Line2D(
    [0], [0], color="#e58213", marker="h", linewidth=0, markersize=11, alpha=0.8
)
line = Line2D([0], [0], color="black", linewidth=0.6, markersize=1, alpha=1)
cmap_gradient = get_gradient_cmap()

## labels
if country_code == "FR":
    labels = [
        # "Potentiel emplacement station 20 spots",
        "Emplacement station 10 spots",
        "Score de demande latente",
        "Limite commune",
    ]
else:
    labels = ["Bikeshare station", "Bikeshare potential", "City limit"]
handles = [hexagon1, cmap_gradient, line]  # hexagon1,

add_legend(
    ax, labels, handles, title="Réseau VLS Communauté de l'Auxerrois", location="best"
)
add_city_names(city, font_size=7)
add_north_arrow(ax)
add_scalebar(ax)

print("plotted map elements")


# basemap
cx.add_basemap(
    ax,
    crs=city.crs,
    # source =#cx.providers.Esri.WorldGrayCanvas ,#cx.providers.OpenStreetMap.Mapnik,#cx.providers.Stamen.TonerLite, #cx.providers.Stamen.Toner ,
    source=cx.providers.Stamen.Terrain,
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

# %% tags=[]
m = selected_sites.explore(
    "site_rank",
    height="100%",
    width="100%",
)
city.explore(m=m, style_kwds={"fill": False})

# %% [markdown]
# ## Exports

# %%
city_boundary = city.copy()  # .query("city_name.isin(@curr_city)").copy()
city_boundary.geometry = city_boundary.boundary

# %%
temp_dict = {
    "city": city_boundary,
    "stations": selected_sites[
        ["geometry", "site_rank", "city_name", f"hex{hex_resolution}"]
    ],
}
temp_dict_20 = {
    "stations_20": selected_sites_20[
        ["geometry", "site_rank", "bike_spots", "city_name", f"hex{hex_resolution}"]
    ],
}
temp_dict_10 = {
    "stations_10": selected_sites_10[
        ["geometry", "site_rank", "bike_spots", "city_name", f"hex{hex_resolution}"]
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

# %% [raw]
# #half the Auxerre stations to map the precise location for Francesco
# save_gdf({
#     "stations_20": selected_sites[7:-14][
#         ["geometry", "site_rank", "bike_spots", "city_name", f"hex{hex_resolution}"]
#     ],
# }, data_dir_out, place)

# %%
save_gdf(temp_dict_20, data_dir_out, place)
save_gdf(temp_dict_10, data_dir_out, place)

# %%
from IPython.display import Markdown as md

if country_code == "FR":
    output = (
        f"{place} - Proposition Réseau VLS\n\n"
        # f"Proposition de maillage pour un réseau VLS dans les villes de {target_city_name} et {(', ').join([x for i,x in enumerate(city_subset) if x!=target_city_name])}.\n\n"
        "Emplacement approx. stations VLS\n\n"
        "Limites communes\n\n"
    )
else:
    output = (
        f"{place} - Bike Share Network Proposal\n\n"
        # f"Proposal for a bike share network in the city of {target_city_name} and {(', ').join([x for i,x in enumerate(city_subset) if x!=[target_city_name]])}.\n\n"
        "Approx. bike station location\n\n"
        "City boundary"
    )

md(output)

# %% [markdown]
# ## Slack message

# %%
from IPython.display import Markdown as md


def print_kpi_predictions(proposed_bikes, kpi_df):
    text = []
    city_name = kpi_df["LIBGEO"].values[0]
    rounded_bikes = round(kpi_df["rounded_bikes"].values.sum())
    bike_fleet_delta = (
        (rounded_bikes - proposed_bikes) / 2 + proposed_bikes
    ) / rounded_bikes  # used to adjust KPIs based on bikes

    if proposed_bikes < (rounded_bikes / 3):
        text += [
            f"WARNING: chosen fleet size is very small compared to optimal fleet size: {rounded_bikes}, this might generate an underperforming service as well as inaccurate predictions"
        ]
    elif proposed_bikes > (rounded_bikes * 2):
        text += [
            f"WARNING: chosen fleet size is very large compared to optimal fleet size :{rounded_bikes}, such a large fleet might not be necessary and cause inaccurate predictions"
        ]

    for scenario_type, scenario_type_var in [
        ("Pessimistic", "clients_low_est"),
        ("Optimistic", "clients_high_est"),
    ]:
        text += [
            f"\n__{scenario_type} scenario for {city_name} with {proposed_bikes} proposed bikes__\n",
            f"Expected users per year: {round(bike_fleet_delta*kpi_df[scenario_type_var].values.sum()):,}",
        ]
        for price_type, price_type_var in [
            ("Low", "low"),
            ("Medium", "med"),
            ("High", "high"),
        ]:
            n_trips = round(
                bike_fleet_delta
                * kpi_df[
                    f"{scenario_type[0]}_{price_type_var}_price_trips_Y"
                ].values.sum()
            )
            text += [f"\n{price_type} pricing estimation\n"]
            text += [
                f"- Expected trips per year: {n_trips:,}",
                f"- Expected bike daily rotation: {round(n_trips/365/proposed_bikes,2):,}",
                f"- Expected km per year: {round(bike_fleet_delta*kpi_df[f'{scenario_type[0]}_{price_type_var}_km_Y'].values.sum()):,}",
                f"- Expected saved CO2 kg per year: {round(bike_fleet_delta*kpi_df[f'{scenario_type[0]}_{price_type_var}_kg_CO2_saved'].values.sum()):,}",
            ]
    return md("\n".join(text))


print_kpi_predictions(220, market_vls)

# %%
if country_code == "FR":
    output = (
        f"__{place} Demand Assessment__\n\n"
        ">__Périmètre:__\n\n"
        f"{(', ').join(market_vls.LIBGEO.unique())}\n\n"
        ">__Taille de flotte__ (détail par ville ci-dessous)\n\n"
        # f"- VLD : {market_vld.bikes.sum()} vélos\n\n"
        f"- VLS : {market_vls.rounded_bikes.sum()} vélos / {market_vls.stations.sum()} stations\n\n"
        ">__Détail interdistance__ (en mètres)\n\n"
        ">__Couverture population__ (approximative)\n\n"
        ">__Cartes__\n\n"
        f"- Cartes statiques ci-jointes\n\n"
        f"- Carte Google MyMaps (lien)\n\n"
    )
else:
    output = (
        f"__{place} Demand Assessment__\n\n"
        ">__Area of study:__\n\n"
        # f"{(', ').join(target_cities)}\n\n"
        ">__Fleet sizing__ (detail by city below)\n\n"
        # f"- VLD : {market_vld.bikes.sum()} bikes\n\n"
        f"- VLS : {market_vls.rounded_bikes.sum()} bikes / {market_vls.stations.sum()} stations\n\n"
        ">__Distance between stations__ (in meters)\n\n"
        ">__Population coverage__\n\n"
        ">__Maps__\n\n"
        f"- Static map attached\n\n"
        f"- Google MyMaps (link)\n\n"
        # f"{selected_sites['dist_to_next_site'].describe()[['count','mean','min','max']]}"
    )

md(output)

# %%
with pd.option_context("float_format", "{:.2f}".format):
    print(
        weighted_dist_to_next_site(selected_sites, weights=[1, 0])
        .describe()[["count", "mean", "min", "max"]]
        .to_frame("")
    )

# %%
with pd.option_context("float_format", "{:.2f}".format):
    print(
        weighted_dist_to_next_site(
            selected_sites[selected_sites["city_name"] == "Brest"], weights=[1, 0]
        )
        .describe()[["count", "mean", "min", "max"]]
        .to_frame("")
    )

# %%
for buffer in [400, 1000]:

    covered_pop = get_pop_coverage(
        selected_sites,
        geo_data_dict["POPULATION_ALL_PER_POINT"].data,
        local_crs,
        buffer=buffer,
    )
    if country_code == "FR":
        print(
            f"Population à moins de {buffer}m d'une station : {covered_pop} soit environ {round(covered_pop/total_pop*100)}% de la population"
        )
    else:
        print(
            f"People living less than {buffer}m from a station : {covered_pop} being around {round(covered_pop/total_pop*100)}% of total population"
        )

# %%
