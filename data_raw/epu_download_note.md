# epu download instructions (manual)

- open the epu data page:
  - https://www.policyuncertainty.com/all_country_data.html

- download the dataset that contains the series you plan to use:
  - europe epu (recommended as a general proxy)
  - or a nearby country/region if needed

- save the file as:
  - data_raw/epu.csv

- expected minimal columns (you can rename them later):
  - date (or year-month)
  - epu_index

- note about frequency:
  - epu is typically monthly
  - later, we will align monthly epu to daily fx dates by forward-filling the last available monthly value
