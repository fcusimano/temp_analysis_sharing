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
# # Brest AO
#
# ### Setup

# %%
# from matplotlib.lines import Line2D
# import contextily as cx
# from matplotlib.legend_handler import HandlerLine2D
# from matplotlib.patches import Patch
# import fiona

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

# %% [markdown]
# ## City selection

# %%
place = 'Brest'
country_code = 'FR'

local_crs = LocalCRS[country_code].value
data_dir_in, data_dir_out = set_up_directories(country_code, place)

# %% [markdown]
# <font color='#8db0f0'><b>France</b></font> 

# %%
target_cities_df=get_city_selection_df(target_city_name=[place])
target_cities_df

# %% [markdown] tags=[]
# ## Fleet sizing
# #### __Long-term rental scheme (VLD)__

# %% [markdown]
# <font color='#8db0f0'><b>France</b></font> 

# %%
market_vld = get_vld_fleet(target_cities_df, batches=5)
print('Recommended long bike rental fleet size by city:')
print(f"{len(market_vld)} cities and {int(market_vld.bikes_lower.sum())}-{int(market_vld.bikes_upper.sum())} bikes")
print_fleet_df(market_vld[['city','bikes',]])

# %%
# Increasing fleet size
market_vld_2 = reset_fleet(market_vld, vld=True)
market_vld_2['bikes_max_2'] = my_round(market_vld_2['bikes_max_2'],10).astype(int)
print('VLD Market Sizing (increased)')
print_fleet_df(market_vld_2.rename(columns={'bikes_max_2':'Bikes'})[['city','Bikes']])

# %%
market_vld = market_vld_2

# %% [markdown]
# #### __Bikeshare (VLS)__

# %% [markdown]
# <font color='#8db0f0'><b>France</b></font> 

# %%
market_vls = get_vls_fleet(target_cities_df, batches=10)
print('VLS Market Sizing')
print_fleet_df(market_vls[['city','bikes','bikes_min','bikes_max','stations']])

# %%
market_vls_2 = reset_fleet(market_vls,vls=True)
print('VLS Market Sizing (increased)')
print_fleet_df(market_vls_2[['city','pop','bikes','bikes_min_2','bikes_max_2','stations']])

# %% [raw]
# market_vls = market_vls_2

# %% [markdown] tags=[]
# ## Geography

# %% [markdown]
# <font color='#8db0f0'><b>France</b></font> 

# %%
specific_place_name = get_specific_place_name(target_cities_df)
city = get_city(specific_place_name, local_crs)

# %%
city.plot(column='city_name',legend=True,figsize=(6,6),legend_kwds={'bbox_to_anchor': (1.5, 1)}).axis('off');

# %%
city.area *1e-6

# %% [markdown]
# ### H3

# %%
hex_resolution = 10

#Generate H3 hexagons
base = get_h3_fill(city,hex_resolution,local_crs)

#Plot
ax = base.plot(figsize=(8,8))
city.plot(ax=ax, color='None').axis('off');

# %% [markdown] tags=[]
# ## External data

# %%
# ox config
ox.config(log_file=True, log_console=False, use_cache=True, timeout=5000)

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
    ('hospitals', ['Centre de Soins et de Réadaptation', "Clinique du Grand Large", "Clinique de l'Iroise", "Clinique Pasteur-Lanroze", "Hôpital Psychiatrique de Bohars", "Hôpital d'Instruction des Armées Clermont-Tonnerre", "Polyclinique Keraudren", "Centre Winnicott", "Ty-Yann centre de soins et de réadaptation"]),
    # ('high_schools',['Bilborough College','Nottingham College - Basford']),
    ("universities", ["Institut d'administration des entreprises de Bretagne Occidentale", "Service Universitaire de Médecine Préventive", "Institut de Géoarchitecture","Institut national supérieur du professorat et de l'éducation", "Institut travail éducatif et social"])
    # ('townhalls',['Mairie de Gaubert'])
]

for category, names_to_remove in to_remove_tuple:
    geo_data_dict[f'DESTINATIONS_{category}_PER_POINT'].data = geo_data_dict[f'DESTINATIONS_{category}_PER_POINT'].data.query("~name.isin(@names_to_remove)")

# %%
geo_data_dict.update(overall_h3_aggregation(geo_data_dict.copy(), base))
geo_data_dict.keys()

# %% [markdown] tags=[]
# ### Overall scoring

# %%
import functools as ft

def compute_PT_score(base_df: gpd.GeoDataFrame, score_var: str) -> np.array:
    """ "
    Compute percentile rank of h3 hexagon based on a score element such as transit stops.
    Force 0 when score element is not present in hexagon.
    """
    PT_score = (base_df[score_var] / base_df.area_km2).rank(
        pct=True, na_option="bottom"
    )
    return np.where(base_df[score_var] == 0, 0, PT_score)

def overall_score(
    base: gpd.GeoDataFrame,
    gdf_list: List[gpd.GeoDataFrame],
    hex_col: str,
    weights: List[int] = [],
) -> gpd.GeoDataFrame:
    """
    Compute final score for each hexagon from the aggregation of the differente OSM elements,
    with the option to set custom weights for specific elements (equal weights by default).
    """
    weights = [10] * (len(gdf_list)) if len(weights) == 0 else weights
    gdf_list.insert(0, base)  # add base to the list to start merging
    return (
        ft.reduce(
            lambda left, right: pd.merge(left, right, on=hex_col, how="left"), gdf_list
        )
        .assign(
            score=lambda x: x[[col for col in x.columns if "PT" in col]]
            .mul(weights)
            .sum(axis=1)
        )
        .pipe(lambda x: x.loc[:, ~x.columns.str.contains("area_km2")])
    )



# %%
geo_data_dict["LANES_ALL_PER_HEX"].data

# %%
geo_data_dict["LANES_ALL_PER_HEX"].data = geo_data_dict["LANES_ALL_PER_HEX"].data.assign(PT_lanes=lambda x: compute_PT_score(x, "LANES"))
geo_data_dict["TRANSIT_ALL_PER_HEX"].data = geo_data_dict["TRANSIT_ALL_PER_HEX"].data.assign(PT_transit=lambda x: compute_PT_score(x, "TRANSIT"))
geo_data_dict["OFFICES_ALL_PER_HEX"].data = geo_data_dict["OFFICES_ALL_PER_HEX"].data.assign(PT_offices=lambda x: compute_PT_score(x, "OFFICES"))
geo_data_dict["DESTINATIONS_ALL_PER_HEX"].data = geo_data_dict["DESTINATIONS_ALL_PER_HEX"].data.assign(PT_destinations=lambda x: compute_PT_score(x, "DESTINATIONS"))


# %%
geo_data_dict["TRANSIT_ALL_PER_HEX"].data

# %%
final_score = overall_score(base, [
    geo_data_dict["LANES_ALL_PER_HEX"].data,
    geo_data_dict["TRANSIT_ALL_PER_HEX"].data,
    geo_data_dict["OFFICES_ALL_PER_HEX"].data,
    geo_data_dict["DESTINATIONS_ALL_PER_HEX"].data],
    'hex10'
)

# %%

# %%
geo_data_dict.update(overall_h3_scoring(geo_data_dict,base.copy(),weights=[1,1.2,0.3,0,1,1]))

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
final_score.plot(
    cmap='magma',
    column='score',
    edgecolor='none',
    legend=True,
    figsize=(12,8),
).axis('off');

# %%
geo_data_dict['POPULATION_ALL_PER_HEX'].data.POPULATION.sum()

# %% [markdown] tags=[]
# ## Tweak fleet sizing

# %%
market_vls 

# %%
print_fleet_df(market_vls[['city','bikes','stations']])
print('')
market_vls_final = (
    market_vls
    .assign(
        bikes=lambda x: x.bikes, #myround(x.bikes*1.1,10).astype(int),
        stations=lambda x: compute_stations_given_bikes(x.bikes).astype(int),
    )
)

print_fleet_df(market_vls_final[['city','bikes','stations']])

# %% [raw]
# VLD Market Sizing (increased)
#                      city  Bikes
# 1                   Brest    720
# 2                Guipavas     40
# 3      Plougastel-Daoulas     30
# 4                Plouzané     30
# 5       Le Relecq-Kerhuon     30
# 6                 Guilers     20
# 7                Gouesnou     20
# 8                  Bohars     10

# %%
# market_vls_final=market_vls_final.drop(1)
# OPTION 1
market_vls_final.loc[market_vls_final['city']=='Brest','stations'] = 10
market_vls_final.loc[market_vls_final['city']=='Guipavas','stations'] = 2
market_vls_final.loc[market_vls_final['city']=='Guilers','stations'] = 1
market_vls_final.loc[market_vls_final['city']=='Plougastel-Daoulas','stations'] = 2
market_vls_final.loc[market_vls_final['city']=='Gouesnou','stations'] = 1
market_vls_final.loc[market_vls_final['city']=='Le Relecq-Kerhuon','stations'] = 1
market_vls_final.loc[market_vls_final['city']=='Plouzané','stations'] = 2
market_vls_final.loc[market_vls_final['city']=='Bohars','stations'] = 1

# OPTION 2
market_vls_final.loc[market_vls_final['city']=='Brest','stations'] = 35
market_vls_final.loc[market_vls_final['city']=='Guipavas','stations'] = 6
market_vls_final.loc[market_vls_final['city']=='Guilers','stations'] = 4
market_vls_final.loc[market_vls_final['city']=='Plougastel-Daoulas','stations'] = 5
market_vls_final.loc[market_vls_final['city']=='Gouesnou','stations'] = 4
market_vls_final.loc[market_vls_final['city']=='Le Relecq-Kerhuon','stations'] = 6
market_vls_final.loc[market_vls_final['city']=='Plouzané','stations'] = 6
market_vls_final.loc[market_vls_final['city']=='Bohars','stations'] = 4

print_fleet_df(market_vls_final[['city','stations']])

# %% [markdown]
# # Bike parking

# %%
bike_parking = gpd.read_file(os.path.join(data_dir_in, "stationnement-velos.geojson"))
bike_parking = bike_parking.to_crs(local_crs)
bike_parking.sample(3)

# %%
bike_parking.nb_places.sum()

# %%
top_10 = pd.read_excel(os.path.join(data_dir_in, "top10.xlsx"))
top_10 = gpd.GeoDataFrame(
    top_10, geometry=gpd.points_from_xy(top_10.longitude, top_10.latitude), crs=WGS84_CRS)
top_10 = top_10.assign(hex9= lambda x: x.apply(
                    lambda x_: lat_lng_to_h3(x_, hex_resolution), axis=1
                )
        )
top_10 = top_10.to_crs(local_crs)
top_10

# %% [markdown] tags=[]
# ## Station selection

# %%
poi_hexes = get_poi_hexes(geo_data_dict, hex_resolution,skip_categories=['high_schools','bus_stations'])
poi_hexes = filter_osm_hexes(poi_hexes, base, market_vls_final)

# %%
manual_hexes = []

# 20 stations (14 fixed stations + 6 stations based on usage)
manual_coords = [
    # 5 Brest
    [48.38766011912694, -4.480587195467345], # train station
    [48.39024051126938, -4.4852274303995605], # place Libérté
    [48.38184441267685, -4.4919711257025705], # musée naval marine
    [48.38432174588501, -4.499508387500098], # recouvrance (top 8)
    [48.38621386317079, -4.493101113717749], # Jean Moulin (téléférique)
    # 3 Plouzané et le Relecq-Kerhuon
    [48.36026757407196, -4.570782495260652], # Plouzané - technopole
    [48.38170408098816, -4.620871353526477], # Plouzané - centre
    [48.397274551638255, -4.422350640257679], # le Relecq-Kerhuon - stadiumpark
    # 3 Guipavas et Plougastel-Daoulas
    [48.43519121105121, -4.400940868382235], # Guipavas - mairie/centre
    [48.432038769770806, -4.405264779424196],# Guipavas -centre sud
    [48.37427405082104, -4.369374965132724], # Plougastel-Daoulas - centre
    # 3 Gouesnou, Guilers et Bohars
    [48.4512496387069, -4.466215049403001], # Gouesnou - marie/centre
    [48.424546332146164, -4.558502623237856], # Guilers - place de la liberation/centre
    [48.42886824312454, -4.5147467196792865], # Bohars - mairie/centre    
]
# manual_coords_hexes  = get_manual_selection_hexes(manual_coords, hex_resolution)
# manual_hexes.extend(manual_coords_hexes)


# 70 stations (50 fixed stations + 20 stations based on usage)
top_10_hexes = top_10[~top_10['Nom de la Station'].isin(['Liberté Quartz','Recouvrance','Moulin Blanc'])].hex9.unique()

manual_coords_hexes  = get_manual_selection_hexes(manual_coords, hex_resolution)
additional_manual_coords = [[48.42850307807448, -4.569866025252759], # Guilers - Leclerc
                            [48.42532432344397, -4.5507218576162805], # Guilers - collège
                            [48.41698630318297, -4.438873819161911], # Guipavas -les portes de Brest
                            [48.41301148895175, -4.450946095371982],# Guipavas - eglise notre dame (sud ouest)
                            [48.405211718531284, -4.484641578308741], # brest cite scolaire
                            [48.402309787928516, -4.466766399282231], # brest place de strasbourg
                            [48.41931895435784, -4.467823257586287], # brest université nord
                           ] 
additional_manual_coords_hexes  = get_manual_selection_hexes(additional_manual_coords, hex_resolution)

manual_hexes.extend(manual_coords_hexes)
manual_hexes.extend(top_10_hexes)
manual_hexes.extend(option_1_hexes)
manual_hexes.extend(additional_manual_coords_hexes)

len(manual_hexes)


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
for c in market_vls_final.city.unique():  
        print('\n====',c.upper(), '====')
        df_bs_sites,test = clustered_sites(
            geo_data_dict['FINAL_SCORE_ALL_PER_HEX'].data[geo_data_dict['FINAL_SCORE_ALL_PER_HEX'].data['city_name']==c],
            target= market_vls_final[market_vls_final['city']==c]['stations'].values[0],
            score_col='score',
            hex_col=f"hex{hex_resolution}",
            neighbors_list=neighbors_list,
            rank_col='site_rank',
            ascending=False,
            k=hex_k_mapper[hex_resolution],
            hex_resolution = hex_resolution,
            # max_distance=2000,
            manual_selection=manual_hexes
        )
        selected_sites = selected_sites.append(df_bs_sites)
print("")
print(selected_sites.selected.value_counts(dropna=False))
selected_sites = selected_sites.query("selected == True")

# %% [markdown]
# ## Maps

# %%
print_fleet_df(market_vls_final[['city','bikes','stations']])

# %%
# Check if stations placed == stations expected
assert_station_selection(selected_sites, market_vls_final)

# %%
# check if we have double stations
extra_stations = compute_stations_given_bikes(market_vls_final.bikes.sum()) - (len(selected_sites))
print(f"{round((extra_stations /len(selected_sites)),2)*100}% of the statsions have 20 bike spots.")

# %% [markdown]
# ## Authorized freefloating area

# %%
freefloating_area = selected_sites.copy()
freefloating_area.geometry = selected_sites.buffer(500)
freefloating_area = freefloating_area.dissolve(by='city_name').reset_index()
freefloating_area.geometry = freefloating_area.convex_hull
freefloating_area.loc[1,'geometry'] = city.query("city_name=='Brest'").geometry.values[0] # No area in Brest

freefloating_area.plot(column='city_name',legend=True,legend_kwds={'bbox_to_anchor': (1.5, 1)}).axis('off');

# %%
bike_parking.columns

# %%
bike_parking['nb_places'].plot.box(figsize=(6,10))

# %%
bike_parking['nb_places'].describe()

# %%
bike_parking.query("nb_places >100").T

# %%
bike_parking.dom_prive.value_counts()

# %%
bike_parking_subset = bike_parking.query("""
    dom_prive == 'Public' and \
    5 <= nb_places < 100
    """)
bike_parking_subset.describe()

# %%
bike_parking_subset.nb_places.sum()

# %%
bike_parking_subset = gpd.sjoin(bike_parking_subset, freefloating_area)
len(bike_parking_subset)

# %%
bike_parking_subset.nb_places.sum() /2

# %% [raw]
# option_1_hexes = selected_sites.hex10.unique()
# len(option_1_hexes)

# %% [raw]
# option_2_additional_hexes = selected_sites.query("not hex10.isin(@option_1_hexes)").hex10.unique()

# %%
selected_sites.query("hex10.isin(@option_1_hexes)").shape

# %% [raw]
# add_Quicksand_font()
#
# not_city = get_city_negative(city)
#
# fig, ax = plt.subplots(1, 1, figsize=(10,10))
# _ = ax.axis('off')
#
# # reset zoom level (due to not_city box)
# xlim, ylim = get_map_limits(city)
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
#
# not_city.plot(
#     ax=ax,
#     color='white',
#     edgecolor='none',
#     alpha=.3,
# )
#
# city.plot(
#     ax=ax,
#     color='none',
#     edgecolor='grey',
#     linewidth=.4,
#     alpha=1
# )
#
# selected_sites.query("hex10.isin(@option_1_hexes)").plot(
#     ax=ax,
#     color='red',
#     edgecolor='none',
#     alpha=.85,
# )
#
# selected_sites.query("not hex10.isin(@option_1_hexes)").plot(
#     ax=ax,
#     color='orange',
#     edgecolor='none',
#     alpha=.85,
# )
#
# freefloating_area.plot(
#     ax=ax,
#     color="None",
#     edgecolor='black',
# )
# bike_parking_subset.plot(
#     ax=ax,
#     color='black',
#     edgecolor='None',
#     markersize=1
# )
#
# # LEGEND
#
# add_city_names(city,font_size=7)
#
#
# # basemap 
# cx.add_basemap(
#     ax,
#     crs=city.crs,
#     source =cx.providers.Esri.WorldGrayCanvas,# cx.providers.OpenStreetMap.Mapnik,# #cx.providers.Stamen.TonerLite, #cx.providers.Stamen.Toner 
#     attribution=None,
#     attribution_size=2
# )
#
# # plt.savefig(os.path.join(data_dir_out,'Brest_maps_v2',"virtual_stations_area.png"),dpi=300,facecolor='w', bbox_inches='tight',)

# %%
add_Quicksand_font()

not_city = get_city_negative(city)

fig, ax = plt.subplots(1, 1, figsize=(10,10))
_ = ax.axis('off')

# reset zoom level (due to not_city box)
xlim, ylim = get_map_limits(city)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

final_score.query("score > score.quantile(.1)").plot(
    ax=ax,
    column='score',
    cmap='Blues',
    edgecolor='none',
    alpha=.6
)

not_city.plot(
    ax=ax,
    color='white',
    edgecolor='none',
    alpha=.3,
)

city.plot(
    ax=ax,
    color='none',
    edgecolor='grey',
    linewidth=.4,
    alpha=1
)

selected_sites.query("hex10.isin(@option_1_hexes)").plot(
    ax=ax,
    color='red',
    edgecolor='none',
    alpha=.85,
)

selected_sites.query("not hex10.isin(@option_1_hexes)").plot(
    ax=ax,
    color='orange',
    edgecolor='none',
    alpha=.85,
)

# freefloating_area.plot(
#     ax=ax,
#     color="None",
#     edgecolor='black',
# )
bike_parking_subset.plot(
    ax=ax,
    color='black',
    edgecolor='None',
    markersize=1
)

# scalebar
ax.add_artist(ScaleBar(1,location='lower right',box_alpha=0))

# north arrow
x, y, arrow_length = 0.05, 0.98, 0.08
ax.annotate('N',xy=(x, y),xytext=(x, y-arrow_length),arrowprops=dict(arrowstyle="wedge,tail_width=0.5,shrink_factor=0.4",facecolor='black'),ha='center',va='center',fontsize=20,xycoords=ax.transAxes)

# LEGEND
## legend elements
hexagon1 = Line2D([0], [0], color='red', marker='h', linewidth=0, markersize=9, alpha=0.8)
hexagon2 = Line2D([0], [0], color='orange', marker='h', linewidth=0, markersize=9, alpha=0.8)

point = Line2D([0], [0], color='black', marker='.', linewidth=0, markersize=5, alpha=1)

line = Line2D([0], [0], color='grey', linewidth=.6, markersize=1, alpha=1 ,label='a')
cmap_gradient = get_gradient_cmap()

## labels
labels = ["Stations chargeantes (option 1)","Stations chargeantes \nsupplémentaires (option 2)","Stations virtuelles",'Score de demande latente', 'Limites communes'] 
handles = [hexagon1,hexagon2,point,cmap_gradient,line]

add_legend(ax, labels, handles,title='Brest Métropole', location='lower left')
add_city_names(city,font_size=7)


# basemap 
cx.add_basemap(
    ax,
    crs=city.crs,
    source =cx.providers.Esri.WorldGrayCanvas,# cx.providers.OpenStreetMap.Mapnik,# #cx.providers.Stamen.TonerLite, #cx.providers.Stamen.Toner 
    attribution=None,
    attribution_size=2
)

# plt.savefig(os.path.join(data_dir_out,'Brest_maps_v2',"Brest_Métropole_option_2.png"),dpi=300,facecolor='w', bbox_inches='tight',)

# %%
# map per city
for curr_city in city.city_name.unique():
    
    not_city = get_city_negative(city.query("city_name == @curr_city"))
    
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    _ = ax.axis('off')

    # reset zoom level (due to not_city box)
    xlim, ylim = get_map_limits(city.query("city_name == @curr_city"))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    final_score.query("city_name_x == @curr_city").query("score > score.quantile(.4)").plot(
        ax=ax,
        column='score',
        cmap='Blues',
        edgecolor='none',
        alpha=.6
    )

    not_city.plot(
        ax=ax,
        color='white',
        edgecolor='none',
        alpha=.3,
    )

    city.query("city_name == @curr_city").plot(
        ax=ax,
        color='none',
        edgecolor='grey',
        linewidth=.4,
        alpha=1
    )

    selected_sites.query("city_name == @curr_city").query("hex10.isin(@option_1_hexes)").plot(
        ax=ax,
        color='red',
        edgecolor='none',
        alpha=.85,
    )
    selected_sites.query("city_name == @curr_city").query("not hex10.isin(@option_1_hexes)").plot(
        ax=ax,
        color='orange',
        edgecolor='none',
        alpha=.85,
    )


    gpd.sjoin(bike_parking_subset[bike_parking.columns],city.query("city_name == @curr_city")).plot(
        ax=ax,
        color='black',
        edgecolor='None',
        markersize=4
    )

    # scalebar
    ax.add_artist(ScaleBar(1,location='lower right',box_alpha=0))

    # north arrow
    x, y, arrow_length = 0.05, 0.98, 0.08
    ax.annotate('N',xy=(x, y),xytext=(x, y-arrow_length),arrowprops=dict(arrowstyle="wedge,tail_width=0.5,shrink_factor=0.4",facecolor='black'),ha='center',va='center',fontsize=20,xycoords=ax.transAxes)

    # LEGEND
    ## legend elements
    hexagon1 = Line2D([0], [0], color='red', marker='h', linewidth=0, markersize=9, alpha=0.8)
    hexagon2 = Line2D([0], [0], color='orange', marker='h', linewidth=0, markersize=9, alpha=0.8)
    point = Line2D([0], [0], color='black', marker='.', linewidth=0, markersize=5, alpha=1)

    line = Line2D([0], [0], color='grey', linewidth=.6, markersize=1, alpha=1 ,label='a')
    cmap_gradient = get_gradient_cmap()

    ## labels
    labels = ["Stations chargeantes (option 1)","Stations chargeantes \nsupplémentaires (option 2)","Station virtuelle",'Score de demande latente', 'Limite commune'] 
    handles = [hexagon1,hexagon2,point,cmap_gradient,line]

    add_legend(ax, labels, handles,title=curr_city, location='best')


    # basemap 
    cx.add_basemap(
        ax,
        crs=city.crs,
        source =cx.providers.Esri.WorldGrayCanvas,# cx.providers.OpenStreetMap.Mapnik,# #cx.providers.Stamen.TonerLite, #cx.providers.Stamen.Toner 
        attribution=None,
        attribution_size=6
    )

    # plt.savefig(os.path.join(data_dir_out,'Brest_maps_v2',f"{curr_city.replace(' ','_')}_option_2.png"),dpi=300,facecolor='w', bbox_inches='tight',)

# %%
# Additional virtual stations in option 1 from stations placed in option 2 
# but only if they do not already have virtual station nearby

new_virtual_stations = selected_sites.query("not hex10.isin(@option_1_hexes)").copy()
new_virtual_stations.geometry = new_virtual_stations.centroid

area_with_no_new_virtual_stations_needed = bike_parking_subset[bike_parking.columns].copy()
area_with_no_new_virtual_stations_needed.geometry = area_with_no_new_virtual_stations_needed.buffer(100)
hexes_with_no_new_virtual_stations_needed = gpd.sjoin(new_virtual_stations, area_with_no_new_virtual_stations_needed).hex10.unique()

new_virtual_stations = new_virtual_stations.query("not hex10.isin(@hexes_with_no_new_virtual_stations_needed)")

# %%
len(new_virtual_stations)  
# len(bike_parking_subset)

# %% [raw]
# bike_parking_option_1 = new_virtual_stations.append(bike_parking_subset)

# %% tags=[]
add_Quicksand_font()

not_city = get_city_negative(city)

fig, ax = plt.subplots(1, 1, figsize=(10,10))
_ = ax.axis('off')

# reset zoom level (due to not_city box)
xlim, ylim = get_map_limits(city)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

final_score.query("score > score.quantile(.1)").plot(
    ax=ax,
    column='score',
    cmap='Blues',
    edgecolor='none',
    alpha=.6
)

not_city.plot(
    ax=ax,
    color='white',
    edgecolor='none',
    alpha=.3,
)

city.plot(
    ax=ax,
    color='none',
    edgecolor='grey',
    linewidth=.4,
    alpha=1
)

selected_sites.query("hex10.isin(@option_1_hexes)").plot(
    ax=ax,
    color='red',
    edgecolor='none',
    alpha=.85,
)

bike_parking_option_1.plot(
    ax=ax,
    color='black',
    edgecolor='None',
    markersize=1
)

# top_10.plot(
#     ax=ax,
#     color='white',
#     edgecolor='red',
#     markersize=1
# )

# scalebar
ax.add_artist(ScaleBar(1,location='lower right',box_alpha=0))

# north arrow
x, y, arrow_length = 0.05, 0.98, 0.08
ax.annotate('N',xy=(x, y),xytext=(x, y-arrow_length),arrowprops=dict(arrowstyle="wedge,tail_width=0.5,shrink_factor=0.4",facecolor='black'),ha='center',va='center',fontsize=20,xycoords=ax.transAxes)

# LEGEND
## legend elements
hexagon1 = Line2D([0], [0], color='red', marker='h', linewidth=0, markersize=9, alpha=0.8)
point = Line2D([0], [0], color='black', marker='.', linewidth=0, markersize=5, alpha=1)

line = Line2D([0], [0], color='grey', linewidth=.6, markersize=1, alpha=1 ,label='a')
cmap_gradient = get_gradient_cmap()

## labels
labels = ["Stations chargeantes (option 1)","Stations virtuelles",'Score de demande latente', 'Limite commune'] 
handles = [hexagon1,point,cmap_gradient,line]

add_legend(ax, labels, handles,title='Brest Métropole', location='lower left')
add_city_names(city,font_size=7)


# basemap 
cx.add_basemap(
    ax,
    crs=city.crs,
    source =cx.providers.Esri.WorldGrayCanvas,# cx.providers.OpenStreetMap.Mapnik,# #cx.providers.Stamen.TonerLite, #cx.providers.Stamen.Toner 
    attribution=None,
    attribution_size=6
)

# plt.savefig(os.path.join(data_dir_out,'Brest_maps_v2',"Brest_Métropole_option_1.png"),dpi=300,facecolor='w', bbox_inches='tight',)

# %%
# map per city
for curr_city in city.city_name.unique():
    
    not_city = get_city_negative(city.query("city_name == @curr_city"))
    
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    _ = ax.axis('off')

    # reset zoom level (due to not_city box)
    xlim, ylim = get_map_limits(city.query("city_name == @curr_city"))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    final_score.query("city_name_x == @curr_city").query("score > score.quantile(.4)").plot(
        ax=ax,
        column='score',
        cmap='Blues',
        edgecolor='none',
        alpha=.6
    )

    not_city.plot(
        ax=ax,
        color='white',
        edgecolor='none',
        alpha=.3,
    )

    city.query("city_name == @curr_city").plot(
        ax=ax,
        color='none',
        edgecolor='grey',
        linewidth=.4,
        alpha=1
    )

    selected_sites.query("city_name == @curr_city").query("hex10.isin(@option_1_hexes)").plot(
        ax=ax,
        color='red',
        edgecolor='none',
        alpha=.85,
    )

    gpd.sjoin(bike_parking_option_1[bike_parking.columns],city.query("city_name == @curr_city")).plot(
        ax=ax,
        color='black',
        edgecolor='None',
        markersize=4
    )

    # scalebar
    ax.add_artist(ScaleBar(1,location='lower right',box_alpha=0))

    # north arrow
    x, y, arrow_length = 0.05, 0.98, 0.08
    ax.annotate('N',xy=(x, y),xytext=(x, y-arrow_length),arrowprops=dict(arrowstyle="wedge,tail_width=0.5,shrink_factor=0.4",facecolor='black'),ha='center',va='center',fontsize=20,xycoords=ax.transAxes)

    # LEGEND
    ## legend elements
    hexagon1 = Line2D([0], [0], color='red', marker='h', linewidth=0, markersize=9, alpha=0.8)
    point = Line2D([0], [0], color='black', marker='.', linewidth=0, markersize=5, alpha=1)

    line = Line2D([0], [0], color='grey', linewidth=.6, markersize=1, alpha=1 ,label='a')
    cmap_gradient = get_gradient_cmap()

    ## labels
    labels = ["Stations chargeantes (option 1)","Stations virtuelles",'Score de demande latente', 'Limite commune'] 
    handles = [hexagon1,point,cmap_gradient,line]

    add_legend(ax, labels, handles,title=curr_city, location='best')


    # basemap 
    cx.add_basemap(
        ax,
        crs=city.crs,
        source =cx.providers.Esri.WorldGrayCanvas,# cx.providers.OpenStreetMap.Mapnik,# #cx.providers.Stamen.TonerLite, #cx.providers.Stamen.Toner 
        attribution=None,
        attribution_size=6
    )

    # plt.savefig(os.path.join(data_dir_out,'Brest_maps_v2',f"{curr_city.replace(' ','_')}_option_1.png"),dpi=300,facecolor='w', bbox_inches='tight',)

# %%
# sanity check
m = selected_sites.drop('k_ring_2',axis=1).explore('site_rank',height='100%',width='100%')
city.explore(m=m,style_kwds={'fill':False})
top_10.explore(m=m)

# %% [markdown]
# # New request: should we remove the stations that are within 300 meters of a physical station?

# %%
for BUFFER in [100,200,300]: # meters
    buffer_check_gdf = selected_sites.copy()
    buffer_check_gdf.geometry  = selected_sites.centroid.buffer(BUFFER)
    buffer_check_gdf = buffer_check_gdf.geometry.unary_union


    within_ = bike_parking_subset[bike_parking_subset.geometry.within(buffer_check_gdf)]
    outside_ = bike_parking_subset[~bike_parking_subset.geometry.within(buffer_check_gdf)]

    print(f"{len(outside_)} virtual stations outside a {BUFFER}m buffer from physical stations out of {len(bike_parking_subset)}")

# %%
outside_.explore()

# %%
within_.shape

# %%
161+94

# %% [markdown] tags=[]
# ## Exports
# #### Google MyMaps

# %%
selected_sites.query("hex10.isin(@manual_coords_hexes)").shape

# %%
fiona.supported_drivers['KML'] = 'rw'
selected_sites.query("hex10.isin(@option_1_hexes)")[['geometry','site_rank','city_name',f"hex{hex_resolution}"]].to_file(os.path.join(data_dir_out,'Brest_option_A.kml'), driver='KML')

city_boundary = city.copy()#.query("city_name.isin(@curr_city)").copy()
city_boundary.geometry = city_boundary.boundary
city_boundary[['geometry','city_name']].to_file(os.path.join(data_dir_out,place+'_city.kml'), driver='KML')

# %%
selected_sites.query("not hex10.isin(@option_1_hexes)")[['geometry','site_rank','city_name',f"hex{hex_resolution}"]].query("not hex10.isin(@manual_coords_hexes)").to_file(os.path.join(data_dir_out,'Brest_option_B.kml'), driver='KML')


# %%
bike_parking_subset[['geometry','nb_places']].to_file(os.path.join(data_dir_out,'Brest_stations_virtuelles.kml'), driver='KML')


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
# #### Slack message

# %%

# %%
if country_code == 'FR':
    output = (
        f"__{place} Demand Assessment__\n\n"
        ">__Périmètre:__\n\n"
        # f"{(', ').join(city_subset)}\n\n"
        ">__Taille de flotte__ (détail par ville ci-dessous)\n\n"
        f"- VLD : {market_vld_final.bikes.sum()} vélos\n\n"
        f"- VLS : {market_vls_final.bikes.sum()} vélos / {market_vls_final.stations.sum()} stations\n\n"
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
        f"- VLD : {market_vld.bikes.sum()} bikes\n\n"
        f"- VLS : {market_vls_final.bikes.sum()} bikes / {market_vls_final.stations.sum()} stations\n\n"
        ">__Distance between stations__ (in meters)\n\n"
        ">__Population coverage__\n\n"
        ">__Maps__\n\n"
        f"- Static map attached\n\n"
        f"- Google MyMaps (link)\n\n"
        # f"{selected_sites['dist_to_next_site'].describe()[['count','mean','min','max']]}"
    )

md(output)

# %% [markdown] tags=[]
# #### Distance between stations

# %%
selected_sites.query("hex10.isin(@option_1_hexes)").shape

# %%
bike_parking_option_1.shape

# %%

# %%
bike_parking_option_1.nb_places.fillna(5).describe()

# %%
with pd.option_context('float_format', '{:.2f}'.format): 
    print(weighted_dist_to_next_site(pd.concat([selected_sites.query("hex10.isin(@option_1_hexes)"),bike_parking_option_1]), weights=[1,0])
          .describe()[['count','mean','min','max']]
          .to_frame(''))

# %%
# selected_sites.query("not hex10.isin(@option_1_hexes)")

# %%
with pd.option_context('float_format', '{:.2f}'.format): 
    print(weighted_dist_to_next_site(selected_sites.query("hex10.isin(@option_1_hexes)"), weights=[1,0])
          .describe()[['count','mean','min','max']]
          .to_frame(''))

# %% [markdown]
# #### Population coverage

# %%
for buffer in [400, 1000]:
    if country_code == 'FR': 
        print(f"Population à moins de {buffer}m d'une station : {get_pop_coverage(selected_sites, geo_data_dict['POPULATION_ALL_PER_POINT'].data, local_crs, buffer=buffer)}")
    else :
        print(f"People living less than {buffer}m from a station : {get_pop_coverage(selected_sites, geo_data_dict['POPULATION_ALL_PER_POINT'].data, local_crs, buffer=buffer)}")

# %% [raw]
# print("VLD market sizing")
# print(tabulate(add_total_row(market_vld_final[['city','bikes']]),headers='keys',tablefmt = 'simple'))

# %% [raw]
# print("VLS market sizing")
# print(tabulate(add_total_row(market_vls_final[['city','bikes','stations']]),headers='keys',tablefmt = 'simple'))

# %%

# %%
