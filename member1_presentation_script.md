# Member 1 Presentation Script (about 2 minutes)

Hi, I handled the data preparation and exploratory analysis for our NYC Street Tree Health Predictor.

We are using the NYC Open Data 2015 Street Tree Census. Each row is one street tree, and our target variable is health, which has three categories: Good, Fair, and Poor. After cleaning, our working sample has 49,994 usable records.

For cleaning, I dropped rows missing the target or core fields like tree diameter, borough, or species. I standardized missing categorical values to Unknown so the app would not break, and I filtered unrealistic diameter values outside 0 to 100 inches.

I also created several features for the model and app: problem_count, has_problem, dbh_group, species_top15_or_other, and binary flags for root, trunk, and branch problems. These features are useful because they are understandable to users and easy for the model to process.

The main EDA finding is that the classes are imbalanced: most trees are labeled Good, while Fair and Poor are much smaller groups. Because of that, our model evaluation should not rely on accuracy alone. We should also report macro-F1 and a confusion matrix.

The visualizations show health distribution, borough differences, common species, problem status, diameter groups, and a map sample. These charts support the data visualization page of the app and help explain what the model is learning from.

One limitation is that this is a 2015 snapshot, not live tree health data. Also, the model can identify patterns, but it cannot prove what causes tree health problems. We should describe the app as an educational and triage-support tool, not an official maintenance decision system.
