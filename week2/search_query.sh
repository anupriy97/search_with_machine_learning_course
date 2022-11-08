# Adding synonyms in index (for "name" field) definitely increased the recall

python utilities/query.py --query nespresso # Returned 8 results
python utilities/query.py --query nespresso --synonyms # Returned 1135 results

python utilities/query.py --query earbuds # Returned 1205 results
python utilities/query.py --query earbuds --synonyms # Returned 8050 results

python utilities/query.py --query dslr # Returned 2837 results
python utilities/query.py --query dslr --synonyms # Returned 4606 results