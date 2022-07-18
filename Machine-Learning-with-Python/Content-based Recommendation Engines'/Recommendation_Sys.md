## What are recommender systems?
- Recommender systems capture the pattern of peoples' behavior and use it to predict what else they might want or like.

### Applications:
- What to buy?
    - E-commerce, books, movies, beer, shoes.
- Where to eat ?
- Which job to apply to ?
- Who you should be friends with ?
    - LinkedIn, Facebook,...
- Personalize your experience on the web.
    - News platforms, news personalization.

### Advantages of recommender systems:
- Broader exposure.
- Possibility of continual usage or purchase of products.
- Provides better experience.

### Two types of recommender systems:
- Content-Based: Show me more of the same of what I've liked before.
- Collaborative Filtering: Tell me what's popular among my neighbors, I also might like it.

### Implementing recommender systems:
- Memory-based: 
    - Uses the entire user-item database to generate a recommendation.
    - Uses statistical techniques to approximate users or items eg: Pearson Correlation, Cosine Similarity, Euclidean Distance, etc.
- Model-based:
    - Develops a model of users in an attempt to learn their preferences.
    - Models can be created using Machine Learning Techniques like regression, clustering, classification, etc.

#### Content-based Recommender systems:
A Content-based recommendation system tries to recommend items to users based on their profile. The user's profile revolves around that user's preferences and tastes. It is shaped based on user ratings, including the number of times that user has clicked on different items or perhaps even liked those items. The recommendation process is based on the similarity between those items. Similarity or closeness of items is measured based on the similarity in the content of those items. 

#### Collaborative Filtering systems:
Collaborative filtering is based on the fact that relationships exist between products and people's interests. Many recommendation systems use collaborative filtering to find these relationships and to give an accurate recommendation of a product that the user might like or be interested in. Collaborative filtering has basically two approaches: user-based and item-based.

- Item-Based Collaborative Filtering: Based on items similarity.
- User-Based Collaborative Filtering: Based on the user similarity and neighborhood.

#### Challenges of collaborative filtering:
- Data Sparsity:
    - Users in general rate only a limited number of item.
- Cold Start:
    - Difficulty in recommendation to new users or new items.
- Scalability:
    - Increase in number of users or items.

