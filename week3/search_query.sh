# Adding synonyms in index (for "name" field) definitely increased the recall
# Adding query category filter improved the precision (even while using explicit sort (such as price, recency))
# Adding query category boost improved the precision but using any explicit sort (such as price, recency) cancels the benefit of the boosting

python utilities/query.py --query nespresso # Returned 8 results
python utilities/query.py --query nespresso --synonyms # Returned 1135 results
python utilities/query.py --query nespresso --query_category_filter # Returned 8 results
python utilities/query.py --query nespresso --query_category_boost 10 # Returned 2819 results

# earbuds classified to "All Headphones" category with score 0.69
python utilities/query.py --query earbuds # Returned 1205 results
python utilities/query.py --query earbuds --synonyms # Returned 8050 results
python utilities/query.py --query earbuds --query_category_filter # Returned 344 results
python utilities/query.py --query earbuds --query_category_boost 10 # Returned 1542 results

# dslr classified to "DSLR Body & Lens" category with score 0.76
python utilities/query.py --query dslr # Returned 2837 results
python utilities/query.py --query dslr --synonyms # Returned 4606 results
python utilities/query.py --query dslr --query_category_filter # Returned 109 results
python utilities/query.py --query dslr --query_category_boost 10 # Returned 2837 results

# ipad classified to "iPad" category with score 0.6
python utilities/query.py --query ipad # Returned 5657 results
python utilities/query.py --query ipad --query_category_filter # Returned 48 results

# laptop classified to "PC Laptops" category with score 0.56
python utilities/query.py --query laptop # Returned 7255 results
python utilities/query.py --query laptop --query_category_filter # Returned 1492 results