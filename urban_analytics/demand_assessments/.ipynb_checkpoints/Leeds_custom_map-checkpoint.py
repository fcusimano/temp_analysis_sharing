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
# # Leeds Custom Map
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
place = 'Leeds'
country_code = 'GB'

local_crs = LocalCRS[country_code].value
data_dir_in, data_dir_out = set_up_directories(country_code, place)

# %%
local_crs

# %%
target_city_name = place


# %% [markdown]
# ## City selection

# %%
target_city_name = place

# # using our datasets
fr_vls = load_fleet_info(vls=True)
target_cities_df = fr_vls.query("country_id == @country_code and city in @place").copy()

# target_cities_df.loc[:,'pop'] = 300_000 

target_cities_df
# np.sort(fr_vls.query("country_id == @country_code").city.unique()).tolist()[:20]

# using citypopulation.de
city_info = target_cities_df[['city','pop','area_km2']].pipe(reset_city_typology)
city_info

# %% [markdown]
# ## Fleet sizing
#

# %% [markdown]
# ### VLS Market

# %% tags=[]
fr_vls = load_fleet_info(vls=True)

vls_stats = (
        # fr_vls.query("country_id == @country_code")
        fr_vls.query("region == 'Europe'")
        # fr_vls.query("country_id == 'FR'")
        .groupby("typology")
        .first()[["min_bikes_km2",
                "avg_bikes_km2",
                "max_bikes_km2",
                "min_bikes_10000ppl",
                "max_bikes_10000ppl",]]
    )

# %%
df = city_info.merge(vls_stats, on="typology", how="left")
df["bikes_min_1"] = (df["area_km2"] * df["min_bikes_km2"]) / 4
df["bikes_max_1"] = (df["area_km2"] * df["max_bikes_km2"]) / 4

df["bikes_min_2"] = (df["pop"] / 10_000) * df["min_bikes_10000ppl"]
df["bikes_max_2"] = (df["pop"] / 10_000) * df["max_bikes_10000ppl"]

df["bikes_min"] = df[["bikes_min_1", "bikes_min_2"]].mean(axis=1)
df["bikes_max"] = df[["bikes_max_1", "bikes_max_2"]].mean(axis=1)
df = df.assign(
    bikes=lambda x: my_round(x[["bikes_min", "bikes_max"]].mean(axis=1)).astype(int)-20,
    stations=lambda x: round(compute_stations_given_bikes(x.bikes)).astype(int),
)


# %%
market_vls=df.sort_values("pop", ascending=False)
print_fleet_df(market_vls[['city','bikes','stations']])

# %% [raw]
# def reset_fleet(df: pd.DataFrame, vld: bool = False, vls: bool = False) -> pd.DataFrame:
#     """
#     Force reset vld or vls fleet size by updating city typology and recomputing fleet.
#
#     Arguments:
#         df: fleet df obtained from `get_vls_fleet()` or `get_vld_fleet()`
#         vld: set to true if reseting vld fleet
#         vls: set to true if reseting vls fleet
#     """
#     if vld == vls:
#         raise ValueError("Select either vls or vld fleet type")
#     fleet_df = load_fleet_info(vld=vld, vls=vls)
#
#     stats_df = fleet_df.groupby("typology").first()[
#         [
#             "min_bikes_km2",
#             "avg_bikes_km2",
#             "max_bikes_km2",
#             "min_bikes_10000ppl",
#             "max_bikes_10000ppl",
#         ]
#     ]
#     return (
#         df.pipe(reset_city_typology)
#         .merge(stats_df, on="typology", how="left")
#         .assign(
#             bikes_min_1=lambda x: (x.area_km2 * x.min_bikes_km2) / 6,
#             bikes_max_1=lambda x: (x.area_km2 * x.max_bikes_km2) / 6,
#             bikes_min_2=lambda x: (x["pop"] / 7_000) * x.min_bikes_10000ppl,
#             bikes_max_2=lambda x: (x["pop"] / 7_000) * x.max_bikes_10000ppl,
#             bikes_min=lambda x: x[["bikes_min_1", "bikes_min_2"]].mean(axis=1),
#             bikes_max=lambda x: x[["bikes_max_1", "bikes_max_2"]].mean(axis=1),
#             bikes_pop=lambda x: x[["bikes_min_2", "bikes_max_2"]].mean(axis=1),
#             bikes=lambda x: my_round(x[["bikes_min", "bikes_max"]].mean(axis=1)),
#             stations=lambda x: compute_stations_given_bikes(x.bikes),
#         )
#         .astype(
#             {
#                 "bikes_min": int,
#                 "bikes_max": int,
#                 "bikes_pop": int,
#                 "bikes": int,
#                 "stations": int,
#             }
#         )
#         .sort_values("pop", ascending=False)
#     )

# %% [raw]
# market_vls_2 = reset_fleet(market_vls,vls=True)
# print('VLS Market Sizing (increased)')
# print_fleet_df(market_vls_2[['city','pop','area_km2','bikes','bikes_min_2','bikes_max_2','stations']])

# %% [raw]
# market_vls = market_vls_2
# market_vls

# %% [markdown]
# ## Geography

# %%
specific_place_name = ['Leeds'] 
#osm_id= ['R345264'] 

city = get_city(specific_place_name, local_crs, False
               )

# %%
city.plot(column='city_name',legend=True,figsize=(6,8),legend_kwds={'bbox_to_anchor': (1.5, 1)}).axis('off');

# %% [markdown]
# ## H3

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
ox.settings.timeout = 1000

# %%
geo_data_dict = {}
#geo_data_dict.update(get_lanes_data(specific_place_name))
geo_data_dict.update(get_bike_parking_data(specific_place_name))
geo_data_dict.update(get_transit_data(specific_place_name,country_code))
geo_data_dict.update(get_offices_data(specific_place_name))
geo_data_dict.update(get_destinations_data(specific_place_name,country_code))
geo_data_dict.update(get_population_data(city, data_dir_out, place, country_code))

# %%
to_remove_tuple = [
    # ('DESTINATIONS', 'hospitals', ['Priory Hospital Nottingham','Thorneywood']),
    # ('DESTINATIONS','high_schools',["Lycée polyvalent Vincent D'Indy"]),
    # ('DESTINATIONS','townhalls',['Hôtel de Ville']),
    # ('DESTINATIONS','universities',['Mairie de Gaubert']),
    # ('TRANSIT','train_stations',['Vichy']),
    # ('TRANSIT','metro_stations',['Mairie de Gaubert']),
    # ('TRANSIT','bus_stations',['Mairie de Gaubert']),
]

for category, subcategory, names_to_remove in to_remove_tuple:
    geo_data_dict[f'{category}_{subcategory}_PER_POINT'].data = geo_data_dict[f'{category}_{subcategory}_PER_POINT'].data.query("~name.isin(@names_to_remove)")

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
                                            1, # Destinations
                                            #0.5, # Lanes
                                            1, # Bike parking
                                            1, # Offices
                                            0.5, # Population
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
# ## Maps

# %% [raw]
# #manual zoom setting
# ylim=[936385.9069815145, 945016.5310863967]
# xlim= [278326.1024535691, 289414.4840683982]

# %% tags=[]
add_Quicksand_font()

not_city = get_city_negative(city)

fig, ax = plt.subplots(1, 1, figsize=(10,10))
_ = ax.axis('off')

# reset zoom level (due to not_city box)

xlim, ylim = get_map_limits(city)
ax.set_xlim(xlim); ax.set_ylim(ylim)

geo_data_dict['FINAL_SCORE_ALL_PER_HEX'].data.query("score > score.quantile(.95)").plot(
    ax=ax,
    column='score',
    scheme="fisher_jenks_sampled", 
    k=7,
    cmap='Blues',
    edgecolor='none',
    alpha=.8
)

not_city.plot(
    ax=ax,
    color='white',
    edgecolor='none',
    alpha=.4,
)

city.plot(
    ax=ax,
    color='none',
    edgecolor='black',
    linewidth=.4,
    alpha=1
)

#selected_sites.plot(
#    ax=ax,
#    color='red',
#    edgecolor='none',
#)

# LEGEND
## legend elements
hexagon1 = Line2D([0], [0], color='red', marker='h', linewidth=0, markersize=11, alpha=0.8)
line = Line2D([0], [0], color='black', linewidth=.6, markersize=1, alpha=1)
cmap_gradient = get_gradient_cmap()

## labels
if country_code == 'FR':
    labels = ["Potentiel emplacement station",'Score de demande latente', 'Limite commune'] 
else:
    labels = ['Bikeshare station', 'Bikeshare potential', 'City limit']
handles = [hexagon1,cmap_gradient,line]

add_legend(ax, labels, handles,title="Leeds" , location='best')
#add_city_names(city,font_size=9)
add_north_arrow(ax)
add_scalebar(ax)



# basemap 
cx.add_basemap(
    ax,
    crs=city.crs,
    #source =#cx.providers.Esri.WorldGrayCanvas ,#cx.providers.OpenStreetMap.Mapnik,#cx.providers.Stamen.TonerLite, #cx.providers.Stamen.Toner ,
    source =cx.providers.Stamen.Terrain,
    attribution=False,
    attribution_size=7
)

plt.savefig(os.path.join(data_dir_out,f"{place.replace(' ','_')}_stations.png"),dpi=300,facecolor='w', bbox_inches='tight',)

# %%

# %% tags=[]
city.explore(style_kwds={'fill':False})

# %% [markdown]
# ## Exports

# %%
city_boundary = city.copy()#.query("city_name.isin(@curr_city)").copy()
city_boundary.geometry = city_boundary.boundary


# %%
temp_dict = {"city": city_boundary, "stations": selected_sites[['geometry','site_rank','city_name',f"hex{hex_resolution}"]]}


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
        f"- VLD : {market_vld.bikes.sum()} bikes\n\n"
        f"- VLS : {market_vls.bikes.sum()} bikes / {market_vls.stations.sum()} stations\n\n"
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
