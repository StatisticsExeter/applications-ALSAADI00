WITH raw AS (
  SELECT 
    co2_emissions_current AS shortfall, 
    CASE 
      WHEN construction_age_estimated < 1930 THEN 'Old'
      WHEN construction_age_estimated >= 1930 THEN 'Recent'
      ELSE NULL
    END AS age, 
    CASE 
      WHEN number_heated_rooms >= 1 AND number_heated_rooms < 10 THEN number_heated_rooms
      WHEN number_heated_rooms >= 10 THEN 10
      ELSE NULL 
    END AS n_rooms,
    local_authority AS local_authority_code
  FROM energy.energy_certificates
)
SELECT
  shortfall,
  local_authority_code,
  age,
  n_rooms
FROM raw
WHERE shortfall IS NOT NULL
  AND age IS NOT NULL
  AND n_rooms IS NOT NULL;