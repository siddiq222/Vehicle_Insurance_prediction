import json

univariate="The target variable is significantly imbalanced only 12.3 percent of customers responded positively to the cross-sell offer, while high-cardinality features like Region_Code and Policy_Sales_Channel show a long-tail distribution, with a handful of categories accounting for the majority of records."
bivariate="Previously_Insured and Vehicle_Damage emerged as the strongest individual predictors of customer interest customers who are not previously insured respond at ~23%, while those with vehicle damage respond at ~24%, both dramatically higher than the ~12%, baseline. In contrast, Driving_License, Annual_Premium, and Vintage showed almost no relationship with the target."
multi="The combination of Previously_Insured=No and Vehicle_Damage=Yes produces the sharpest signal in the entire dataset a 25.5%, response rate, more than double either feature's individual effect revealing that customer interest is best explained by the interaction between risk history and current insurance status, not any single feature alone."

insights={'univariate':univariate,'bivariate':bivariate,'multivariate':multi}

with open("insights.json", "w") as f:
    json.dump(insights, f, indent=4)