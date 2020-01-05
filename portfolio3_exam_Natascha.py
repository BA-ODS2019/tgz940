#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:27:04 2019

@author: martinejorgensen
"""


# Opgave 3: Ved hjælp af Pandas og request, importere nu jeres valgte datasæt.
import requests
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

api_search_url = 'https://api.smk.dk/api/v1/art/search'


# Vi opretter en 'dictionary' og angiver værdierne for SMKs API's parametre.
# Vi har valgt at vores nøgle-søgeord skal være statuetter. 
params = {
    'keys': 'statuette', 
    'rows': 1000,
}


# For at kunne bruge json beautifier, sørger vi for, at det står i JSON-format.
params['encoding'] = 'json'

# Her anmodes om vores søgning med SMKs API-URL samt de valgte parametre. 
response = requests.get(api_search_url, params=params)

# Således kan vi printe URL'en: 
print('Here\'s the formatted url that gets sent to the smk API:\n{}\n'.format(response.url)) 


json = response.json()


df = json_normalize(json['items'])

# Vi tjekker kolonnerne i head, her kan vi se, at der er 49 kolonner
df.head()


# Her kan vi se alle data-typerne
df.dtypes


# Fjerner kolonner
df.drop(['iiif_manifest','object_history_note','image_native','image_thumbnail','has_image','related_objects','work_status','production_dates_notes','credit_line','current_location_name','content_person','distinguishing_features','alternative_images','image_mime_type', 'image_iiif_id', 'image_iiif_info', 'image_width','image_height', 'image_size', 'image_cropped', 'image_orientation','exhibitions', 'labels'], inplace=True, axis=1)



# Navngiver kolonner 
df.rename(columns={'created':'reg_year'}, inplace=True)
df.rename(columns={'object_names':'Type'}, inplace=True)
df.rename(columns={'number_of_parts':'number'}, inplace=True)
df.rename(columns={'acquisition_date_precision':'acqdate'}, inplace=True)



# Her kan vi se, at der nu er 26 kolonner
print(df.head())


# Formen på dataframen: 537 rækker og 26 kolonner
print(df.shape)











# Opgave 4: Udtræk og beregn 

# Finder NaN værdier/missing values og returnerer summen. 
# Således kan vi se, at der forekommer celler med manglende værdier
print(df.isnull().sum()) 


# Antal af værker, der er registreret som public domain. 
df['public_domain'].sum()


# Kan se alle årstallene 
alleårstal = []
for i in range(len(df['acqdate'])):
    alleårstal.append(df['acqdate'][i][0:4])
    alleårstal.sort()
    
from collections import Counter
Counter(alleårstal)
 


# Kun unikke årstal i datasættet
unikkeårstal =[]  
def unique(alleårstal): 
    x = np.array(alleårstal) 
    unikkeårstal.extend(np.unique(x)) 

unique(alleårstal)
unikkeårstal # Svar: 111



# Fordelingen af forskellige typer af værker inden for nøgleordet, statuettes. 
df['Type'].value_counts() 

# Fordelingen af værker registreret som public domæne. False 292 || True 245. Fleste er ikke.
df['public_domain'].value_counts() 

# Hvor mange er public i alt
df['public_domain'].sum()

# Optællinger
df['acqdate'].str.extract(r'^(\d{4})', expand=False).value_counts() # vi udtrækker de specifikke årstal og foretager en optælling
df['materials'].value_counts()
df['techniques'].value_counts(normalize=True)









# Opgave 5: Ny dataframe med value_counts 

# Vi har lavet en procentvis optælling af acquisition date
optæl_årstal = df['acqdate'].str.extract(r'^(\d{4})', expand=False).value_counts(normalize=True).mul(100).round(1) # får kun optællingerne
nydfdate = pd.DataFrame(optæl_årstal)

# Her har vi genereret en dataframe med årstal og optællingen 
opdeltdf = pd.DataFrame({'år': nydfdate.index, 'forekomst(%)': optæl_årstal})









# Opgave 6: Visualisering af den nye dataframe. Dette fremvises i Jupytor Lab.

# Altair biblioteket importeres. 
import altair as alt

# Ved x har vi sat år, som er den horisontale linje 
# Ved y har vi sat forekomst i procent, dvs. optællingen, som vi indsatte i en dataframe, og vises i den vertikale linje 
alt.Chart(opdeltdf).mark_line(point=True).encode(
        x='år',
        y='forekomst(%)',
        tooltip= [alt.Tooltip('år'), alt.Tooltip('forekomst(%)')] # for hover effect 
    ).properties(width=2500, height=300)



