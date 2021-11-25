import requests
import json
api_key = "7ae9458e891125ad33f584367f999d05"

movie_db = {
    "results":[
        
    ]
}
template = {
    "title":[],
    "features":[{
        "genre_ids":[],
        "production_companies":[],
        "release_date":[],
        "vote_average":[],
        "vote_count":[],
        "budget":[],
        "profit":[]
    }
    ]
   
}

for i in range(1,501):

    response = requests.get("https://api.themoviedb.org/3/discover/movie?api_key="+ api_key+"&include_adult=false"+"&page="+str(i))
    result = response.json()

    j = 0
    for movie in result['results']:
        template = {
            "title":[],
            "features":[{
                "genre_ids":[],
                "production_companies":[],
                "release_date":[],
                "vote_average":[],
                "vote_count":[],
                "budget":[],
                "profit":[],
                "star_popularity":[]
            }]  
        }

        try:
            movie_id = result['results'][j]['id']
            movie_response = requests.get("https://api.themoviedb.org/3/movie/" + str(movie_id) +"?api_key="+ api_key+"&language=en-US")
            movie_result = movie_response.json()
            #movie_result = json.dumps(movie_result)
            #print(movie_result)
            #print(movie_result['adult'])
            template['title'] = movie_result['title']
            #template['features'][0]['genre_ids'] = movie_result['genre_ids']
            template['features'][0]['release_date'] = movie_result['release_date']
            for x in range(0,len(movie_result['production_companies'])):
                template['features'][0]['production_companies'].append(movie_result['production_companies'][x]['id'])
            template['features'][0]['vote_average'] = movie_result['vote_average']
            template['features'][0]['vote_count'] = movie_result['vote_count']
            template['features'][0]['budget'] = movie_result['budget']
            template['features'][0]['profit'] = movie_result['revenue'] - movie_result['budget']


            credit_response = requests.get("https://api.themoviedb.org/3/movie/" + str(movie_id) +"/credits?api_key="+api_key+"&language=en-US")
            credit_result = credit_response.json()
            # movie_result = json.dumps(credit_result)
            # print(movie_result)
            template['features'][0]['star_popularity']= credit_result['cast'][0]['popularity']

            movie_db['results'].append(template)    
        except KeyError:
            print("KeyError")
        except IndexError:
            print("IndexError")
        j+=1


db=json.dumps(movie_db, indent=4)
with open("movie_db.json", "w") as outfile:
    outfile.write(db)



