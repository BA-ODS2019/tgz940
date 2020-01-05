#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:19:33 2019

@author: martinejorgensen
"""


# OPGAVE (1)  Åben filen i en tekst-editor og se på indholdet. 

# OPGAVE (a)
# I filen ‘titanic.csv’, som er åbnet med Numbers, kan man anskue følgende datatyper, ‘interger’ og ‘float’, 'string'. 
# Den førstnævnte angiver, at værdien er et heltal, den næste et decimaltal og sidstnævne et tekststykke. 
# Endvidere er hele datasættet opdelt i 8 kolonner: ‘Survived’, ‘Pclass’, ‘Name’, ‘Sex’, ‘Siblings/Spouses Aboard’, ‘Parents/Children Aboard’ og ‘Fare’. 
# De overlevende/omkomne bliver angivet som henholdsvis '1'/‘0’ (intergers).
# Dernæst kan man også se opdelingen af klasser, 1, 2 og 3 (interger), samt navne på passagerne og deres køn (strings). 
# Derudover kan man også se deres alder, antal søskende, ægtefæller og forældre (interger) og billetprisen (float).


# OPGAVE (b)
# Når vi gennemgår Numbers-arket, kan vi se, at der ikke mangler data, eftersom der ikke forekommer tomme celler i dataframen.
# Manglende værdier vil fremstå som tomme celler i Numbers og kategoriseret som ‘NaN’ (Not a Number) i Python.  
# Dette demonstreres også i næste opgave.  






# OPGAVE (2)

# Her importeres pandas
import pandas as pd

# Her læser vi filen titanic med read_csv, da det er en csv-fil.
df_titanic = pd.read_csv('titanic.csv') 

# Dernæst udskriver vi dataframen og kan se, at der i alt eksisterer 887 rækker og 8 kolonner, hvilket svarer til 887 passagerer og 8 kategorier. 
print(df_titanic)


# Nu vil vi bruge describe() til at få et hurtigt overblik over diverse kolonner og generelle statistiske beregninger.
# Her kan vi eksempelvis se, at den højeste pris for en billet er 512,8 og den mindste pris er 0. 
df_titanic.describe() 
 
# klart overblik over alle 8 kategorier i for-loop, der gennemløber dataframen og udskriver kategorien for hvert genenmløb. 
for i in df_titanic.columns: 
    print(i)
    
# Her bruger vi shape() til at illustrer antallet af kolonner og rækker i dataframen.
print(df_titanic.shape) 

# Denne funktion viser antallet af celler. Således kan man se antallet af værdier i hele dataframen
print(df_titanic.size)

# viser datatyper. Funktionen undersøger og definerer forskellige typer data i dataframen og fortæller, at der findes følgende: Int64, object, float64. 
# Dette kan defineres som henholdsvis integer numbers, string, og float. 
# Pandas referer til strings som object. (Kilde: https://pbpython.com/pandas_dtypes.html?fbclid=IwAR06-ND1itRz3rV4UNzihOOLj5IgtVpgZm2Z6FDVsdyVCe4UQhf7jgLb--Y)
print(df_titanic.dtypes)  

# Undersøger om dataframen mangler data. Funktionen stiller spørgsmålet: Er der tomme celler i datasættet. Hvis nej, vil output være False og hvis ja, vil den returnere True.
df_titanic.empty

# Alternativ løsning til at finde missing data. Bruger funktionen isnull() og sum() til at finde missing data. Funktionen gennemår data for hver kolonne, og optæller antallet af True og False, da den ligeledes forsøger at undersøge, om værdien er 0. 
# Såfremt funktionen finder tomme celler, vil den angive dem som True og lave en optælling af dette booleanske udtryk. Hvis funktionen ikke finder tomme celler, vil de blive angivet som False og tilskrevet værdien 0. 
# Derved får vi en værdi ud fra hver kolonne som output, der viser tallet 0, eftersom der ikke forekommer tomme celler i dataframen.  
print(df_titanic.isnull().sum()) 









# OPGAVE (3)

# Denne funktion udskriver antallet af overlevende, da den lægger summen af alle værdierne sammen. 
# 0 er omkomne og 1 er overlevende.
# Således ved vi med sikkerhed at tallet, 342, er antallet af overlevende og er kategoriseret som tallet 1. 
print(df_titanic['Survived'].sum())


# Denne funktion tæller antallet af hver unikke værdi i kolonnen. 
# Her kan vi se både antallet af omkomne (0), 545, og overlevende (1), 342.   
print(df_titanic['Survived'].value_counts())


# Overblik over passasegere på hver klasse. Vi har brugt samme funktion, der tæller antallet af hver unikke værdi.
print(df_titanic['Pclass'].value_counts()) 

# Alternativ måde at se antal rejsende inden for hver klasse med en variabel.
filter_classes=df_titanic['Pclass'].value_counts()
print(filter_classes)

# Gennemsnitsalder med mean()funktionen. 
# Vi angiver en specifik kolonne i dataframen og bruger mean() funktionen. 
average= df_titanic['Age'].mean() 
# Her runder vi tallet til nærmeste heltal.
print(round(average))

# Vi bruger median(), således vi kan se median alderen.
print(df_titanic['Age'].median())

# Vi udskriver kolonnen Name, så vi kan få en oversigt over navnene.
print(df_titanic['Name'])

# Her kan vi få et overblik over den højeste billetpris ved hjælp af max() funktionen. 
print(df_titanic['Fare'].max())

# Her vil output vise den laveste billetpris ved hjælp af min() funktionen. 
print(df_titanic['Fare'].min())

# Her kan vi se gennemsnitsprisen på billetterne ved at bruge mean() igen. 
print(df_titanic['Fare'].mean())








# OPGAVE (4)

# For at undersøge, hvorvidt der forekommer tilfælde, hvor passagererne har samme efternavn, har vi først og fremmest kreeret en ny variabel og dernæst benyttet .str.rsplit() funktionen. 
# Denne funktion splitter kolonnen, Name, i et antal kolonner og opdelinger i forhold til en bestemt seperator. 
# Først kan man definere, hvorvidt kolonnen skal opdeles efter en streng eller regulært udtryk i Pat.
# Dernæst hvor mange opdelinger, der skal forekomme i N, eg. Hvis n = 1, får man 2 opdelinger af dataen, således et navn freemstår som en liste med to elementer: forenavn og efternavn.  
# Slutteligt kan man vælge i expand, om strengene skal udvides i separate kolonner baseret på booleanske udtryk, True og False. 
# Hvis man ikke definerer Pat, vil funktionen, som standard, separere efter mellemrum, hvilket passer med separeringen af navnene. 
# Vi har valgt kun at inkludere N og expand. N er sat til 1 og expand er True. Således får vi to kolonner med opdelingen, fornavne og efternavne. 
# Slutteligt har vi udskrevet antal efternavne (kolonne 1, da kolonne 0 er fornavne) i den nye variabel sammenlagt med antallet af gentagelser, da value_counts tæller de unikke værdiers forekomst. 
# kilde: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.rsplit.html
df_new= df_titanic['Name'].str.rsplit(n=1, expand=True) #kolonner med efternavne
print (df_new) 
df_new[1].value_counts() #navne + antal gentagelser





#En alternativ løsning på opgaven viser alle de efternavne, der går igen i dataframen. 
#Først har vi lavet en variabel, som finder Names og splitter dem ved mellemrum bagfra.
lastnames=df_titanic.Name.str.split(' ').str[-1]
print(lastnames)

# Her kan vi se hele listen over alle efternavnene, alle 887. 
print(lastnames.tolist())

# Her har vi oprettet to tomme liste variable, som vi ønsker at bruge senere.
unique=[]
repeat=[]

# Vi har lavet en for-loop, der tager udgangspunkt i listen med alle efternavnene
# Ved første for-loop runde leder funktionen efter et efternavn. 
# Hvis efternavnet ikke findes i 'unique' listen i forvejen, bliver det tilføjet til den tomme unique liste.
# Ved anden for-loop runde, leder funktionen også først efter om efternavnet findes i 'unique' listen - og hvis det ikke gør, er det stadig unikt og skal tilføjes til den tomme liste
# Derimod hvis efternavnet allerede eksisterer i 'unique', går efternavnet videre til elif-statementet og spørger om efternavnet findes i repeat-listen - på samme måde tilføjes det til repeat listen, hvis navnet ikke er der i forvejen
# Resultatet ender ud med, at alle navne i repeat-listen kun bliver nævnt 1 gang - og vi kan derfor konkludere, at 133 navne ud af de ialt 887, bliver gentaget mere end 1 gang

for i in lastnames:
    if i not in unique:
        unique.append(i)
    elif i not in repeat:
        repeat.append(i)

print(sorted(repeat))
print(len(repeat))









# OPGAVE (5)


# Først har vi genereret en pivot-tabel med klasse, der viser antal rejsende inden for hver klasse (med brug af pandas funktioner.) 
# I dette tabel-format kan man angive hvilken kolonne, der skal vises, og hvilken funktion, der skal behandle den. Dette har vi valgt at drage fordel af i vores tabel. 
# aggfunc udgør den type funktion, man ønsker at anvende, hvilket vi har defineret som count, da vi skal have en optælling af alle rejsende opdelt i klasser. 
# Kilde: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html
class_tabel= df_titanic.pivot_table(columns='Pclass', aggfunc=({'Pclass':'count'}))
print(class_tabel)
#Her bruger vi plot(kind='bar'), således vi også kan få et visuelt output
class_tabel.plot(kind='bar')


#Antal overlevende/omkomne i hver klasse, hvor 0 er omkomne og 1 er overlevende. Funktionen optæller antallet for hver værdi i survived. 
# I første klasse var der 80 omkomne 
# I anden klasse 97
# I trejde klasse 368
# Dermed var trejde klasse den med flest omkomne.
nyplot = df_titanic.groupby(['Pclass', 'Survived'])['Survived'].count()
print(nyplot)
# Her har vi lavet en visuel fremstilling af pivot-tabellen. 
nyplot.plot(kind='bar')



# Her har vi lavet en Pivot tabel, der udelukkende viser fordelingen af overlevende i hver klasse.
# Først har vi lavet en ny dataframe, der består af kolonnerne, Survived og Class. Dernæst har vi brugt .groupby() til at visualisere fordelingen af overlevende samt omkomne på de forskellige klasser.  
# For at generere pivot-tabellet, vil vi anvende den nye dataframe, værdien af de overlevende, som bliver opstille i kolonne i forhold til de tre klasser. 
# Denne pivot-tabel bruger sum(), således vi kan se antallet af overlevende. 
# Funktionen funder summen af alle værdierne, bestående af 0 og 1, og fremviser dem tabellen ud fra hver klasse. 
# Slutteligt har vi inkluderet plot(kind=‘bar'), for at demonstrere fordelingen i et søjlediagram. 

ny_df = df_titanic[['Survived','Pclass']] 
print(ny_df)
ny_df.shape 
tabel1= pd.pivot_table(ny_df, values= 'Survived', columns= 'Pclass', aggfunc= 'sum') #antal overlevende på de forskellige klasser
print(tabel1)
# Her har vi lavet en visuel fremstilling af pivot-tabellen. 
vistabel1 = tabel1.plot(kind='bar')


# Det sidste vi vil demonstrere er, hvor mange der i alt overlevede og omkom, ved brug af en for-loop.
# Den første funktion laver en liste over alle 1'ere og 0'ere - som skal bruges i for-loop funktionen.
count_people=df_titanic['Survived'].tolist()
print(count_people)


#Denne for-loop løber derfor igennem listen 'count_people', og tæller hvor mange der overlevede og hvor mange der døde
survived=0
not_survived=0
for i in count_people:
    if i == 0:
        not_survived = not_survived + 1
    elif i == 1:
        survived = survived + 1
print(survived)
print(not_survived)






