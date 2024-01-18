This is a machine learning project focusing on weather collected by the Saint Louis Science Center ('s weather station)

I got the data from NOAA (.gov)
https://www.ncei.noaa.gov/cdo-web/orders?email=quaaysan@gmail.com&id=3537869
https://docs.google.com/spreadsheets/d/1sLkweI6yhmzwY_4Mpd0yerXh4cPvR39x4XsUUgSiGcg/edit#gid=1827959499

This documentation on what all of this data means
https://www.ncei.noaa.gov/data/daily-summaries/doc/GHCND_documentation.pdf

I'm currently looking to predict weather, temperature high/low, ~~and windspeed~~

Within the data table, this corresponds to:
- PRCP/SNOW/SNWD: Precipitation (or snow and snowdepth)
    - PRCP precipitation measured in inches(?)
    - SNOW snowfall measured in inches
    - SNWD snow depth measured in inches
- TMAX/TMIN: Temperature Highs and Lows
    - TMAX maximum temperature fahrenheit
    - TMIN minimum temperature fahrenheit