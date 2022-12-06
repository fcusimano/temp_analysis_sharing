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

# %% [markdown]
# # Lleida Demand Assessment
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

# %%
place = 'Lleida'
country_code = 'ES'

local_crs = LocalCRS[country_code].value
data_dir_in, data_dir_out = set_up_directories(country_code, place)

# %%
local_crs

# %%
target_city_name = place


# %% [markdown]
# ## City selection

# %%
lugo_dict = {"city": "Lleida","city_name": "Lleida", "lat": 41.6177, "lon": 0.6200, "country_id": "ES", "population":137_856, 
             "timezone": "Europe/Madrid", "country": "Spain", "continent": "Europe", "place_name" : 
             "Lleida,Spain", "typology":"100,000-200,000","region":"Europe","area_km2": 211.7} 
target_cities_df = pd.DataFrame(lugo_dict, index=[0])
city_info = target_cities_df[['city','population','area_km2']].pipe(reset_city_typology)
city_info


# %% [markdown]
# ## Fleet sizing
# ### VLD market

# %% [raw]
# fr_vld = load_fleet_info(vld=True)
#
# vld_stats = (
#     fr_vld
#     .groupby("typology")
#     .first()[["min_bikes_km2",
#               "avg_bikes_km2",
#               "max_bikes_km2",
#               "min_bikes_10000ppl",
#               "max_bikes_10000ppl",]]
#     )
#

# %% [raw]
# df_lugo= (
#     city_info
#     .merge(vld_stats, on="typology", how="left")
#     .assign(
#         bikes_min_1 = lambda x: (x.area_km2 * x.min_bikes_km2) / 3,
#         bikes_max_1 = lambda x: (x.area_km2 * x.max_bikes_km2) / 3,
#         bikes_min_2 = lambda x: (x["pop"] / 10_000) * x.min_bikes_10000ppl,
#         bikes_max_2 = lambda x: (x["pop"] / 10_000) * x.max_bikes_10000ppl,
#         bikes_min = lambda x: x[["bikes_min_1", "bikes_min_2"]].mean(axis=1).astype(int),
#         bikes_max = lambda x: x[["bikes_max_1", "bikes_max_2"]].mean(axis=1).astype(int),
#         bikes = lambda x: my_round(x[["bikes_min", "bikes_max"]].mean(axis=1)).astype(int),
#     )
# )

# %% [raw]
# market_vld=df_lugo.sort_values("pop", ascending=False)
# print_fleet_df(market_vld[['city','bikes','bikes_max']])

# %% [markdown]
# ### VLS Market

# %%
def get_fleet_size(df):
    fleet_sizing_parameters = pd.DataFrame(
        {
            "typology": [
                "0-20,000",
                "20,000-50,000",
                "50,000-100,000",
                "100,000-200,000",
                "200,000-500,000",
                "500,000+",
            ],
            "estimated_km2": [
                2.8,
                5.3,
                6.5,
                7.8,
                11.2,
                49.3,
            ],  # here we have 49 instead of 93 for the 500k+ cities because we had to account the fact that Paris was given 21000k bikes
            # but it doesnt take into consideration that a lot of the bikes are spread on its suburbs
            "estimated_1k": [3.5, 3, 3.5, 4, 6, 8],
        }
    )
    df = pd.merge(df, fleet_sizing_parameters, how="left", on="typology")
    df["avg_bikes_area"] = df["area_km2"] * df["estimated_km2"]
    df["avg_bikes_pop"] = df["population"] * df["estimated_1k"] / 1000
    df = df.assign(
        base_bikes=np.where(
            df.population
            > 2000,  # getting rid of small villages getting large number of bikes from area
            np.clip(
                ((df.avg_bikes_pop * 4 + df.avg_bikes_area) / 5),
                a_min=df.avg_bikes_pop
                * 0.65,  # if bike_area is too small, don't punish towns with low final base_bikes
                a_max=df.avg_bikes_pop
                * 1.5,  # if bike_area is too high, don't reward towns with high final base_bikes (Arles had 1k base_bikes from 5k bike_area)
            ),
            df.avg_bikes_pop,
        )
    )
    return df.drop(columns=["estimated_km2", "estimated_1k"])


# %%
lleida_df= get_fleet_size(city_info)
lleida_df


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
market_vls=get_vls_fleet_vincent(lleida_df)
market_vls

# %%
market_vls["stations"]=65
market_vls["rounded_bikes"]=550

# %% [markdown]
# ## Geography

# %%
specific_place_name = ['Lleida, Spain'] 
#osm_id= ['R345264'] 

city = get_city(specific_place_name, local_crs, False
               )

# %%
city.plot(column='city_name',legend=True,figsize=(6,8),legend_kwds={'bbox_to_anchor': (1.5, 1)}).axis('off');

# %%
manual_boundaries = gpd.read_file('/home/vincentaguerrechariol/Documents/repos/data-notebook-sharing/urban_analytics/demand_assessments/manual_boundaries/lleida_boundary.csv')
manual_boundaries=manual_boundaries.set_crs(WGS84_CRS)
manual_boundaries=manual_boundaries.to_crs(local_crs)
manual_boundaries

# %%
city_manual=city.copy()

# %%
city_manual["geometry"]=manual_boundaries["geometry"]

# %%
city_manual.plot(column='city_name',legend=True,figsize=(6,8),legend_kwds={'bbox_to_anchor': (1.5, 1)}).axis('off');

# %% [markdown]
# ## H3

# %%
hex_resolution = 10

#Generate H3 hexagons
base = get_h3_fill(city_manual,hex_resolution,local_crs)

#Plot
ax = base.plot(figsize=(8,8))
city_manual.plot(ax=ax, color='None').axis('off');

# %% [markdown]
# ## External data

# %%
ox.settings.timeout = 1000

# %%
geo_data_dict = {}
geo_data_dict.update(get_lanes_data(specific_place_name))
geo_data_dict.update(get_bike_parking_data(specific_place_name))
geo_data_dict.update(get_transit_data(specific_place_name,country_code))
geo_data_dict.update(get_offices_data(specific_place_name))
geo_data_dict.update(get_destinations_data(specific_place_name,country_code))
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
    #(
    #    "DESTINATIONS",
    #   "universities",
    #    [
    #        "Lycée Montmajour et Perdiguier",
    #        "École Nationale Supérieure de la Photographie",
    #    ],
    #),
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
geo_data_dict.update(overall_h3_scoring(geo_data_dict,
                                        base.copy(),
                                        weights=[
                                            1, # Transit
                                            0.5, # Destinations
                                            0.5, # Lanes
                                            1, # Bike parking
                                            1, # Offices
                                            0.25, # Population
                                        ]))

# %%
geo_data_dict['FINAL_SCORE_ALL_PER_HEX'].data.describe()

# %%
geo_data_dict['FINAL_SCORE_ALL_PER_HEX'].data.plot(
    cmap='magma',
    column='score',
    edgecolor='none',
    legend=True,
    figsize=(12,8),
).axis('off');

# %%
total_pop= geo_data_dict['POPULATION_ALL_PER_HEX'].data.POPULATION.sum()
total_pop

# %% [markdown]
# ## Station selection

# %%
poi_hexes = get_poi_hexes(geo_data_dict, hex_resolution,
                         skip_categories=['universities',"high_schools", "hospitals", ]
                         )
poi_hexes = filter_poi_hexes(poi_hexes, base, market_vls)

# %%
poi_hexes

# %%
manual_hexes = []
manual_coords = [
    [41.61515094631106, 0.657014137688908],# commercial center
]
manual_coords_hexes  = get_manual_selection_hexes(manual_coords, hex_resolution)
manual_hexes.extend(manual_coords_hexes)
manual_hexes.extend(poi_hexes)
manual_hexes

# %%
hex_k_mapper = {9: 1, 10: 1} # CAREFUL WITH THIS


# %%
# Modified function to remove the option 2 (need to rewrite function when time permits)

def clustered_sites(
    df: gpd.GeoDataFrame,
    target: int,
    score_col: str,
    hex_col: str,
    rank_col: str,
    hex_resolution: int,
    manual_selection: List[str] = [],
    neighbors_list: List[str] = [],
    max_distance: Optional[int] = None,
    ascending: bool = False,
    k: int = 1,
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Selecting H3 sites based on an existing ranking without selecting contiguous sites.

    target: number of sites to select
    score_col: column with the score that will be used for the ranking
    hex_col: column with the H3 index
    rank_col: column with the revised site rankings
    manual_selection: list of hexes to be selected regardless
    max_distance: max distance allowed for a station to be valid (using weighted distance)
    ascending: sorting method for the ranking
    k: k neighbors to be set aside
    #TODO: split into smaller functions, simplify
    """

    # initializing
    neighbors = "k_ring_{}".format(k)
    df[neighbors] = df[hex_col].apply(
        lambda x: [h for h in list(k_ring(x, k)) if h != x]
    )
    df = (
        df.sort_values(score_col, ascending=ascending)
        .reset_index()
        .drop(columns=["index", "level_0"], errors="ignore")
    )
    all_midpoint_selected_hexes = []
    selected_stations_counter = 0
    skiped_stations_counter = 0
    intermediate_stations_counter = 0
    df.loc[:, "selected"] = None

    # check manual hex selection
    if manual_selection is not None:
        for i in list(df.loc[df.loc[:, hex_col].isin(manual_selection), :].index):
            df.loc[i, "selected"] = True
            df.loc[i, "backup"] = False
            selected_stations_counter += 1
            neighbors_list = list(set(neighbors_list + df.loc[i, neighbors]))

    # looping
    for i in range(0, len(df)):
        if target > selected_stations_counter:
            if df.loc[i, "selected"] == True:
                continue
            else:
                if df.loc[i, hex_col] in set(neighbors_list):
                    df.loc[i, "selected"] = False
                    df.loc[i, "backup"] = True
                    continue
                else:
                    df.loc[i, "selected"] = True
                    df.loc[i, "backup"] = False
                    selected_stations_counter += 1
                    neighbors_list = list(set(neighbors_list + df.loc[i, neighbors]))

    # MAX DISTANCE LOGIC - remove the station and add another one that is closer to the network
    if max_distance is not None:
        df.loc[
            df["selected"] == True, "dist_to_next_site"
        ] = weighted_dist_to_next_site(df.loc[df["selected"] == True])
        df1, neighbors_list1 = df.copy(), neighbors_list.copy()

        # OPTION 1
        current_max_distance = df1.loc[
            df1["selected"] == True, "dist_to_next_site"
        ].max()
        while current_max_distance > max_distance:
            print(f"Max distance exceeded! {round(current_max_distance)}m")
            index_to_drop = (
                df1.loc[(df1["selected"] == True) & (~df1[hex_col].isin(manual_selection)), :]
                .sort_values("dist_to_next_site", ascending=False)
                .head(1)
                .index
            )
            skiped_stations_counter += 1
            df1.loc[index_to_drop, "selected"] = False
            df1.loc[index_to_drop, "backup"] = True

            for i in df1.loc[
                (df1["selected"].isna()) & (~df1[hex_col].isin(neighbors_list1)), :
            ].index:
                df1.loc[i, "selected"] = True
                df1.loc[i, "backup"] = False
                neighbors_list1 = list(set(neighbors_list1 + df1.loc[i, neighbors]))
                break
            df1.loc[
                df1["selected"] == True, "dist_to_next_site"
            ] = weighted_dist_to_next_site(df1.loc[df1["selected"] == True])
            current_max_distance = df1.loc[
                df1["selected"] == True, "dist_to_next_site"
            ].max()
        score_df1 = df1[df1["selected"] == True].score.sum()

        if (skiped_stations_counter == 0) & (intermediate_stations_counter == 0):
            print("Max distance not reached")
        else :
            df = df1.copy()
            print(
                f"Skipped {skiped_stations_counter} far away station(s)."
            )
            print(f"New current max distance {round(current_max_distance)}m")

    # rank selected stations
    df.loc[df["selected"] == True, rank_col] = df.loc[
        df["selected"] == True, score_col
    ].rank(method="max", ascending=ascending, na_option="bottom")

    # rank backups and other sites that were not selected, in case we need them
    # df.loc[df['selected']==False,rank_col] = target+df.loc[df['selected']==False,score_col].rank(
    #     method='max',ascending=ascending,na_option='bottom')
    # df[['backup','selected']] = df[['backup','selected']].fillna(False)

    return df.sort_values(rank_col, ascending=True), neighbors_list


# %%
import warnings
warnings.filterwarnings('ignore')

selected_sites = pd.DataFrame()
neighbors_list = []
for c in market_vls.city.unique():  
        print('\n====',c.upper(), '====')
        df_bs_sites,test = clustered_sites(
            geo_data_dict['FINAL_SCORE_ALL_PER_HEX'].data[geo_data_dict['FINAL_SCORE_ALL_PER_HEX'].data['city_name']==c],
            target= (market_vls[market_vls['city']==c]['stations'].values[0]),
            score_col='score',
            hex_col=f"hex{hex_resolution}",
            neighbors_list=neighbors_list,
            rank_col='site_rank',
            ascending=False,
            k=hex_k_mapper[hex_resolution],
            hex_resolution = hex_resolution,
            max_distance=1500, # RESET THIS AT 2000 OR 1000 FOR NEXT DEMAND ASSESSMENT
            manual_selection=manual_hexes,
        )
        selected_sites = selected_sites.append(df_bs_sites)
print("")
print(selected_sites.selected.value_counts(dropna=False))
selected_sites = selected_sites.query("selected == True")

# %% [raw]
# #drop unwanted hexagons
#
# selected_sites.drop(selected_sites[(selected_sites["hex10"] == "8a3925619b0ffff") |
#                                    (selected_sites["hex10"] == "8a392560a797fff") |
#                                    (selected_sites["hex10"] == "8a392560a51ffff") |
#                                    (selected_sites["hex10"] == "8a3925656d1ffff") |
#                                    (selected_sites["hex10"] == "8a3925609767fff") |
#                                    (selected_sites["hex10"] == "8a3925609cc7fff") |
#                                    (selected_sites["hex10"] == "8a392560a757fff")].index, inplace=True)
#                     

# %%
len(selected_sites)

# %%
assert_station_selection(selected_sites, market_vls)

# %%
extra_stations = compute_stations_given_bikes(market_vls.rounded_bikes.sum()) - (len(selected_sites))
print(f"{round((extra_stations /len(selected_sites)),2)*100}% of the statsions have 20 bike spots.")

# %%
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

# %% [markdown]
# ## Maps

# %% [raw]
# #manual zoom setting
# ylim=[936385.9069815145, 945016.5310863967]
# xlim= [278326.1024535691, 289414.4840683982]

# %% tags=[]
add_Quicksand_font()

not_city = get_city_negative(city)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
_ = ax.axis("off")

# reset zoom level (due to not_city box)
#xlim, ylim = get_map_limits(city)

xlim = [954287.0233354534, 965414.4916722955]
ylim = [782945.474740287, 794213.3550075288]

ax.set_xlim(xlim)
ax.set_ylim(ylim)

geo_data_dict["FINAL_SCORE_ALL_PER_HEX"].data.query("score > score.quantile(.90)").plot(
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
if country_code == "FR":
    labels = [
        # "Potentiel emplacement station 20 spots",
        "Potentiel emplacement station 10 spots",
        "Score de demande latente",
        "Limite commune",
    ]
else:
    labels = ["Bikeshare station", "Bikeshare potential", "City limit"]
handles = [hexagon2, cmap_gradient, line]  # hexagon1,

add_legend(ax, labels, handles, title=f"{place} VLS", location="best")
#add_city_names(city, font_size=7)
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

# %%
selected_sites_no_k_ring = selected_sites.copy()
selected_sites_no_k_ring.drop(columns="k_ring_1", inplace=True)

# %% tags=[]
m = selected_sites_no_k_ring.explore(
    "site_rank",
    height="100%",
    width="100%",
)
city.explore(m=m, style_kwds={"fill": False})

# %% [markdown]
# ## Exports

# %%
city_boundary = city.copy()#.query("city_name.isin(@curr_city)").copy()
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
save_gdf(temp_dict_20, data_dir_out, place)
save_gdf(temp_dict_10, data_dir_out, place)


# %%
from IPython.display import Markdown as md

if country_code == 'FR':
    output = (
        f"{place} - Proposition Réseau VLS\n\n"
        # f"Proposition de maillage pour un réseau VLS dans les villes de {target_city_name} et {(', ').join([x for i,x in enumerate(city_subset) if x!=target_city_name])}.\n\n"
        'Emplacement approx. stations VLS\n\n'
        'Limites communes\n\n'
    )
else:
    output = (
        f"{place} - Bike Share Network Proposal\n\n"
        # f"Proposal for a bike share network in the city of {target_city_name} and {(', ').join([x for i,x in enumerate(city_subset) if x!=[target_city_name]])}.\n\n"
        'Approx. bike station location\n\n'
        'City boundary'
    )
    
md(output)

# %% [markdown]
# ## Slack message

# %%
if country_code == 'FR':
    output = (
        f"__{place} Demand Assessment__\n\n"
        ">__Périmètre:__\n\n"
        f"{(', ').join(market_vls.city.unique())}\n\n"
        ">__Taille de flotte__ (détail par ville ci-dessous)\n\n"
        #f"- VLD : {market_vld.bikes.sum()} vélos\n\n"
        f"- VLS : {market_vls.bikes.sum()} vélos / {market_vls.stations.sum()} stations\n\n"
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
        #f"- VLD : {market_vld.bikes.sum()} bikes\n\n"
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
with pd.option_context('float_format', '{:.2f}'.format): 
    print(weighted_dist_to_next_site(selected_sites, weights=[1,0])
          .describe()[['count','mean','min','max']]
          .to_frame(''))

# %%
for buffer in [400, 1000]:
    
    covered_pop = get_pop_coverage(selected_sites, geo_data_dict['POPULATION_ALL_PER_POINT'].data, local_crs, buffer=buffer)
    if country_code == 'FR': 
        print(f"Population à moins de {buffer}m d'une station : {covered_pop} soit environ {round(covered_pop/total_pop*100)}% de la population")
    else :
        print(f"People living less than {buffer}m from a station : {covered_pop} being around {round(covered_pop/total_pop*100)}% of total population")

# %%
