import pandas as pd
file = "C:\\Users\\raths\\OneDrive\\Desktop\\Private\\food_diary_v2_2103.xlsx"
df = pd.read_excel(file, sheet_name="Sheet1")
import numpy as np

class DietInspector:
    def __init__(self, df = df):

        self.data_lst = []
        self.dietmap = {
            "broccoli": {"fodmap": "high", "fructose": 0, "excess_fruc": 0,"sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 0, "id":0, "sorbitol": 0, "protein": 0},
            "quinoa": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":1,"sorbitol": 0, "protein": 6.5},
            "raspberry": {"fodmap": 0, "fructose": 1.68 * 0.33, "excess_fruc": 0.7, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":2,"sorbitol": 0, "protein": 0},
            "potatoes": {"fodmap": 0, "fructose": 0.6, "excess_fruc": 0.2, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":3,"sorbitol": 0, "protein": 3.6},
            "avocado": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 8, "lactose": 0, "gluten": 0, "id":4,"sorbitol": 1, "protein": 0.8},
            "blueberry": {"fodmap": 0, "fructose": 5*0.33, "excess_fruc": 0.2, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":5,"sorbitol": 0,"protein": 0},
            "apple": {"fodmap": 1, "fructose": 8, "excess_fruc": 6, "sugar": 2, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":6,"sorbitol": 1,"protein": 0},
            "tomatoes": {"fodmap": 1, "fructose": 1, "excess_fruc": 0.2, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":7,"sorbitol": 0,"protein": 0},
            "pears": {"fodmap": 1, "fructose": 7, "excess_fruc": 2.5, "sugar": 0.5, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":8,"sorbitol": 0, "protein": 0},
            "almonds": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 5, "lactose": 0, "gluten": 0, "id":9,"sorbitol": 0, "protein": 4},
            "banana": {"fodmap": 1, "fructose": 6, "excess_fruc": 5.5, "sugar": 4, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 0, "id":10,"sorbitol": 0, "protein": 0},
            "oats": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 2.5, "lactose": 0, "gluten": 0, "id":11,"sorbitol": 0, "protein": 8.5},
            "red pepper": {"fodmap": 1, "fructose": 2.6, "excess_fruc": 0.4, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":12,"sorbitol": 0, "protein": 0},
            "white bread": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 0.6, "id":13,"sorbitol": 0, "protein": 6},
            "onion": {"fodmap": 1, "fructose": 1.9, "excess_fruc": -0.4, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 0, "id":14,"sorbitol": 0, "protein": 0},
            "garlic": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 0, "id":15,"sorbitol": 0, "protein": 0},
            "green onion": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":16,"sorbitol": 0, "protein": 0},
            "basil": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":17,"sorbitol": 0, "protein": 0},
            "white pasta": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 2, "fat": 0, "lactose": 0, "gluten": 2, "id":18,"sorbitol": 0, "protein": 6},
            "white pasta half": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 1,"id":19,"sorbitol": 0, "protein": 3},
            "kiwi": {"fodmap": 0, "fructose": 4.3, "excess_fruc": 0.8, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0,"id":20,"sorbitol": 0, "protein": 0},
            "carrots": {"fodmap": 0, "fructose": 0.5, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":21,"sorbitol": 0,"protein": 0},
            "veggie soup": {"fodmap": 1, "fructose": 0.5, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 0, "id":22,"sorbitol": 1, "protein": 0},
            "peas": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 0, "id":23,"sorbitol": 0, "protein": 5},
            "dark chocolate": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 1.5, "fructan": 0, "fat": 4, "lactose": 0.5, "gluten": 0, "id":24,"sorbitol": 0, "protein": 1},
            "falafel": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 1, "id":25,"sorbitol": 0, "protein": 13},
            "whole grain bread": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 0.6, "id":26,"sorbitol": 0, "protein": 13},
            "sweet potato": {"fodmap": 0.5, "fructose": 0.9, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":27,"sorbitol": 1, "protein": 0},
            "chestnut mushroom": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":28,"sorbitol": 1, "protein": 3},
            "green beans": {"fodmap": 0, "fructose": 0.9, "excess_fruc": -0.4, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":29,"sorbitol": 0, "protein": 1.8},
            "endive": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":30,"sorbitol": 0, "protein": 0},
            "peanuts": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 5, "lactose": 0, "gluten": 0, "id":31,"sorbitol": 0, "protein": 5},
            "baked beans": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":32,"sorbitol": 0, "protein": 6},
            "courgette": {"fodmap": 0, "fructose": 1, "excess_fruc": 0, "sugar": 0, "fructan": 0.5, "fat": 0, "lactose": 0, "gluten": 0, "id":33,"sorbitol": 0, "protein": 1.2},
            "strawberry": {"fodmap": 1, "fructose": 2.6, "excess_fruc": 0.4, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":34,"sorbitol": 0, "protein": 0},
            "psyllium": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":35,"sorbitol": 1, "protein": 0},
            "raisins": {"fodmap": 1, "fructose": 10, "excess_fruc": 1, "sugar": 0, "fructan": 0.5, "fat": 0, "lactose": 0, "gluten": 0, "id":36,"sorbitol": 1, "protein": 0.5},
            "asparagus": {"fodmap": 1, "fructose": 1, "excess_fruc": 0.2, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 0, "id":37,"sorbitol": 0, "protein": 2.2},
            "olives": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 2.5, "lactose": 0, "gluten": 0, "id":38,"sorbitol": 0, "protein": 0},
            "apricot": {"fodmap": 1, "fructose": 1.7, "excess_fruc": -0.9, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":39,"sorbitol": 1, "protein": 0},
            "batavia": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":40,"sorbitol": 0, "protein": 0},
            "chicken": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 10, "lactose": 0, "gluten": 0, "id":41,"sorbitol": 0, "protein": 54},
            "steak": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 38, "lactose": 0, "gluten": 0, "id":42,"sorbitol": 0, "protein": 50},
            "bacon": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 10.5, "lactose": 0, "gluten": 0, "id":43,"sorbitol": 0, "protein": 19},
            "crisps": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 17.5, "lactose": 0, "gluten": 0, "id":44,"sorbitol": 0, "protein": 3.5},
            "crisps half": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 8, "lactose": 0, "gluten": 0, "id":45,"sorbitol": 0, "protein": 1.75},
            "turkey": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 7, "lactose": 0, "gluten": 0, "id":46,"sorbitol": 0, "protein": 56},
            "thüringer": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 30, "lactose": 0, "gluten": 0, "id":47,"sorbitol": 0, "protein": 15},
            "sable": {"fodmap": 1, "fructose": 1, "excess_fruc": 0, "sugar": 2, "fructan": 0.2, "fat": 4.5, "lactose": 0.01, "gluten": 0.2, "id":48,"sorbitol": 0, "protein": 0},
            "gf wrap": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 1.5, "fructan": 0, "fat": 6, "lactose": 0, "gluten": 0, "id":49,"sorbitol": 1, "protein": 7},
            "humus": {"fodmap": 1, "fructose": 0.15, "excess_fruc": 0.15, "sugar": 0, "fructan": 1, "fat": 17, "lactose": 0, "gluten": 0, "id":50,"sorbitol": 4},
            "cherries": {"fodmap": 1, "fructose": 5.5, "excess_fruc": 8.3, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":51,"sorbitol": 1, "protein": 0},
            "parmesan cheese": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 3.5, "lactose": 0.5, "gluten": 0, "id":52,"sorbitol": 0, "protein": 4},
            "marble cake muffin": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 16, "fructan": 1, "fat": 10, "lactose": 0.5, "gluten": 1, "id":53,"sorbitol": 0, "protein": 0},
            "lemon muffin": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 10, "fructan": 1, "fat": 10, "lactose": 1, "gluten": 1, "id":54,"sorbitol": 0, "protein": 0},
            "gf bread": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 1.5, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":55,"sorbitol": 0, "protein": 5},
            "white rice": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":56,"sorbitol": 0, "protein": 2.7},
            "red cabbage": {"fodmap": 0.5, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 0, "id":57,"sorbitol": 0, "protein": 0},
            "potatoes dinner": {"fodmap": 0, "fructose": 0.3, "excess_fruc": 0.1, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":58,"sorbitol": 0, "protein": 1.8},
            "green pepper": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":59,"sorbitol": 1, "protein": 0},
            "mäiffelcher":{"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 10, "lactose": 1, "gluten": 1, "id":60,"sorbitol": 0, "protein": 2},
            "zockerwäffelchen": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 4, "fructan": 1, "fat": 5, "lactose": 0.1, "gluten": 0.1, "id":61,"sorbitol": 0, "protein": 2},
            "marble cake": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 16, "fructan": 1, "fat": 10, "lactose": 0.5, "gluten": 1, "id":62,"sorbitol": 0, "protein": 0},
            "chorizo": {"fodmap": 0.5, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 10, "lactose": 0, "gluten": 0, "id":63,"sorbitol": 0, "protein": 8},
            "ham": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 13, "lactose": 0, "gluten": 0, "id":64,"sorbitol": 0, "protein": 6},
            "risotto": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 5, "lactose": 0.5, "gluten": 0, "id":65,"sorbitol": 0, "protein": 2.7},
            "salmon": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 15, "lactose": 0, "gluten": 0, "id": 66,"sorbitol": 0, "protein": 30},
            "gf chicken fingers": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 14, "lactose": 0, "gluten": 0, "id":67,"sorbitol": 0, "protein": 13},
            "eggs x2": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 12, "lactose": 0, "gluten": 0, "id":68,"sorbitol": 0, "protein": 26},
            "egg": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 6, "lactose": 0, "gluten": 0, "id":69,"sorbitol": 0, "protein": 13},
            "tuna": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 2, "lactose": 0, "gluten": 0, "id":70,"sorbitol": 0, "protein": 25},
            "sardines": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 8, "lactose": 0, "gluten": 0, "id":71,"sorbitol": 0, "protein": 25},
            "olive oil drizzle": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 5, "lactose": 0, "gluten": 0, "id":72,"sorbitol": 0, "protein": 0},
            "dressing": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 5, "lactose": 0, "gluten": 0, "id":73,"sorbitol": 0, "protein": 0},
            "butter spread": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 1, "lactose": 0.1, "gluten": 0, "id":74,"sorbitol": 0, "protein": 0},
            "mayo": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 4, "lactose": 0, "gluten": 0, "id":75,"sorbitol": 0, "protein": 0},
            "schnitzel": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 10, "lactose": 0.5, "gluten": 0.25, "id":76,"sorbitol": 0, "protein": 30},
            "cordon bleu": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 20, "lactose": 2, "gluten": 0.25, "id":77,"sorbitol": 0, "protein": 54},
            "dinde roulade": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 15, "lactose": 1, "gluten": 0, "id":78,"sorbitol": 0, "protein": 54},
            "wine": {"fodmap": 0, "fructose": 1, "excess_fruc": 1, "sugar": 0, "fructan": 0, "fat": 0, "lactose": 0, "gluten": 0, "id":79,"sorbitol": 0, "protein": 0},
            "zockerwäffelcher": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 7, "fructan": 1, "fat": 5,"lactose": 0.1, "gluten": 0.1, "id": 80,"sorbitol": 0, "protein": 0},
            "raspberry muffin": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 4, "fructan": 1, "fat": 5,"lactose": 0.1, "gluten": 1, "id": 81,"sorbitol": 0, "protein": 0},
            "tofu burger": {"fodmap": 0, "fructose":0, "excess_fruc":0, "sugar": 0, "fructan": 0, "fat": 10, "lactose": 0, "gluten": 0, "id": 82,"sorbitol": 0, "protein": 10},
            "bolognese sauce": {"fodmap": 0.5, "fructose":1.5, "excess_fruc":0, "sugar": 3.8, "fructan": 0, "fat": 30, "lactose": 0, "gluten": 0, "id": 83,"sorbitol": 0, "protein": 14},
            "olive oil dressing": {"fodmap": 0, "fructose":0, "excess_fruc":0, "sugar": 0, "fructan": 0, "fat": 10, "lactose": 0, "gluten": 0, "id": 84,"sorbitol": 0, "protein": 0},
            "apple tart": {"fodmap": 1, "fructose":2, "excess_fruc":1.5, "sugar": 5, "fructan": 1, "fat": 5, "lactose": 0.1, "gluten": 1, "id": 85,"sorbitol": 1,"protein": 0},
            "cream dressing": {"fodmap": 0, "fructose":0, "excess_fruc":0, "sugar": 0, "fructan": 0, "fat": 5, "lactose": 1, "gluten": 0, "id": 86,"sorbitol": 0, "protein": 0},
            "sorbet": {"fodmap": 1, "fructose": 10, "excess_fruc": 10, "sugar": 0, "fructan": 0, "fat": 0,"lactose": 0, "gluten": 0, "id": 87,"sorbitol": 1, "protein": 0},
            "nutella": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 11, "fructan": 0, "fat": 6,"lactose": 0.4, "gluten": 0, "id": 88,"sorbitol": 0, "protein": 0},
            "turkey geschnetzeltes": {"fodmap": 0.5, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0.5, "fat": 14,"lactose": 0, "gluten": 0.5, "id": 89,"sorbitol": 1, "protein": 54},
            "turkey geschnetzeltes half": {"fodmap":0.5, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0.5, "fat": 7,"lactose": 0, "gluten": 0.25, "id": 90,"sorbitol": 1, "protein": 27},
            "pommerloch salad": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 5,"lactose": 1, "gluten": 0, "id": 91,"sorbitol": 0, "protein": 0},
            "zockerwäffelcherx2": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 14, "fructan": 1, "fat": 10,"lactose": 0.2, "gluten": 0.2, "id": 92,"sorbitol": 0, "protein": 0},
            "boxemännchen": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 6.5, "fructan": 1, "fat": 40,"lactose": 1, "gluten": 1, "id": 93,"sorbitol": 0, "protein": 0},
            "boxemännchen_half": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 3.25, "fructan": 1, "fat": 20,"lactose": 0.5, "gluten": 1, "id": 94,"sorbitol": 0, "protein": 0},
            "jam": {"fodmap": 1, "fructose": 0, "excess_fruc": 3, "sugar": 4, "fructan": 0, "fat": 0,"lactose": 0, "gluten": 1, "id": 95,"sorbitol": 1, "protein": 0},
            "yoghurt": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 7, "fructan": 0, "fat": 3,"lactose": 5, "gluten": 0, "id": 96,"sorbitol": 0, "protein": 0},
            "glace": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 5, "fructan": 0, "fat": 6.5,"lactose": 5, "gluten": 0, "id": 97,"sorbitol": 1, "protein": 0},
            "almond cake": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 10, "fructan": 1, "fat": 6.5,"lactose": 1, "gluten": 1, "id": 98,"sorbitol": 0, "protein": 10},
            "nikki marquisette": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 2, "fructan": 1, "fat": 4.5, "lactose": 0.1, "gluten": 1, "id": 99,"sorbitol": 0, "protein": 1},
            "gf speculoos": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 1.5, "fructan": 0, "fat": 1, "lactose": 0, "gluten": 0, "id": 100,"sorbitol": 0, "protein": 1},
            "gf chicken fingers half": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 7,
                                   "lactose": 0, "gluten": 0, "id": 101, "sorbitol": 0, "protein": 6.5},
            "gf madeleine": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 6, "fructan": 0, "fat": 8,"lactose": 0, "gluten": 0, "id": 102, "sorbitol": 0, "protein": 2.5},
            "pizza sole mio":{"fodmap": 1, "fructose": 1, "excess_fruc": 0.5, "sugar": 0, "fructan": 1, "fat": 10, "lactose": 0, "gluten": 1, "id": 103, "sorbitol": 0, "protein": 4},
            "focaccia": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 10, "lactose": 0, "gluten": 1, "id": 104, "sorbitol": 0, "protein": 6},
            "cotelette": {"fodmap": 1, "fructose": 1, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 10, "lactose": 1, "gluten": 0, "id": 105, "sorbitol": 0, "protein": 20},
            "fischer parisienne sw": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 16, "lactose": 1, "gluten": 1, "id": 106, "sorbitol": 0, "protein": 22},
            "rhubarb tart": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 6, "lactose": 1, "gluten": 0.5, "id": 107, "sorbitol": 0, "protein": 3},
            "radler": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 8, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 1, "id": 108, "sorbitol": 0, "protein": 0},
            "hamburger": {"fodmap": 0, "fructose": 0, "excess_fruc": 0, "sugar": 8, "fructan": 0, "fat": 15, "lactose": 0, "gluten": 0, "id": 109, "sorbitol": 0, "protein": 20},
            "chocolatine": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 10, "fructan": 1, "fat": 15, "lactose": 0.25, "gluten": 1, "id": 110, "sorbitol": 0, "protein": 5},
            "bouillon soup": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 0, "lactose": 0, "gluten": 1, "id": 111, "sorbitol": 0, "protein": 3},
            "fuesend": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 5, "fructan": 1, "fat": 5, "lactose": 0.25, "gluten": 1, "id": 112, "sorbitol": 0, "protein": 1},
            "croissant": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 5, "fructan": 1, "fat": 10, "lactose": 0.25, "gluten": 1, "id": 113, "sorbitol": 0, "protein": 5},
            "chicken ragout": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0.1, "fat": 10, "lactose": 0, "gluten": 0, "id": 114, "sorbitol": 0, "protein": 20},
            "veal ragout sauce": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 0, "fat": 15, "lactose": 0, "gluten": 0, "id": 115, "sorbitol": 0, "protein": 25}
            "chicken schnitzel": {"fodmap": 1, "fructose": 0, "excess_fruc": 0, "sugar": 0, "fructan": 1, "fat": 20, "lactose": 0.1, "gluten": 1, "id": 116, "sorbitol": 0, "protein": 54}

        }

    def assess_episode_10d(self, df = df, days = 10):

        dic = {}
        analytics = self.analytics(df)
        indices = list(df[df["Diarr"] == "Yes"].index)
        for idx in indices:
            cut_off = idx + days
            mean_macros = analytics.loc[idx:cut_off].mean()
            dic[f"{idx} to {cut_off}"] = mean_macros

        return pd.DataFrame(dic)


    def analytics(self, df = df):

        for i, r in df.iloc[:350].iterrows():
            r = r.str.lower()
            r.fillna("none", inplace=True)

            fodmap = 0
            fructose = 0
            excess_fruc = 0
            lactose = 0
            sugar = 0
            fructan = 0
            fat = 0
            gluten = 0
            sorbitol = 0
            protein = 0

            for k in r:
                try:
                    fodmap += self.dietmap[k.lower()].get("fodmap")
                    fructose += self.dietmap[k.lower()].get("fructose")
                    excess_fruc += self.dietmap[k.lower()].get("excess_fruc")
                    lactose += self.dietmap[k.lower()].get("lactose")
                    sugar += self.dietmap[k.lower()].get("sugar")
                    fructan += self.dietmap[k.lower()].get("fructan")
                    fat += self.dietmap[k.lower()].get("fat")
                    gluten += self.dietmap[k.lower()].get("gluten")
                    sorbitol += self.dietmap[k.lower()].get("sorbitol")
                    protein += self.dietmap[k.lower()].get("protein")

                except:
                    continue

            self.data_lst.append((fodmap, fructose, excess_fruc, lactose ,sugar, fructan, fat, gluten, sorbitol, protein))
        self.df = pd.DataFrame(self.data_lst, columns=["fodmap", "fructose", "excess_fruc", "lactose", "sugar", "fructan", "fat", "gluten", "sorbitol", "protein"])
        self.df["insoluble_ratio"] = df["insoluble"] / df["Soluble"]
        self.df["alcohol"] = df["Alc Flag"]

        return self.df

    def abnormality_df(self, df = df, col="Bloating", lag=1):

        selection = df[col]
        data_shift = self.df.shift(-lag)
        selection_df = pd.concat([selection, data_shift], axis = 1)

        return selection_df


    def _nn(self):

        arr_lst = []
        for i, r in df.iloc[:250].iterrows():
            r = r.str.lower()
            r.fillna("none", inplace=True)
            arr = np.zeros(len(self.dietmap))

            for k in r:
                try:
                    arr[self.dietmap[k.lower()].get("id")] = 1

                except:
                    continue

            arr_lst.append(arr)

        return arr_lst

def get_fodmap_score(df):

    fodmap_dic = {"broccoli": "High",
    "quinoa": 0,
    "raspberry": 0,
    "potatoes": 0,
    "avocado": "Medium",
    "blueberry": 0,
    "apple": "High",
    "tomatoes": "High",
    "pears": "High",
    "almonds": "High",
    "banana": "High",
    "oats": 0,
    "red pepper": "High",
    "white bread": "High",
    "onion": "High",
    "garlic": "High",
    "green onion": 0,
    "basil": 0,
    "pine nuts": "High",
    "white pasta": "High",
    "kiwi": 0,
    "carrots": 0,
    "veggie soup": "High",
    "peas": "High",
    "dark chocolate": 0,
    "falafel": "High",
    "whole grain bread": "High",
    "sweet potato": 0,
    "chestnut mushroom": "High",
    "green beans": 0,
    "endive": 0,
    "peanuts": 0,
    "baked beans": "High",
    "courgette": 0,
    "strawberry": "High",
    "psyllium": 0,
    "raisins": "High",
    "asparagus": "High",
    "olives": 0,
    "apricot": "High",
    "batavia": "High",
    "gf wrap": 0,
    "humus": "High",
    "crisps": 0,
    "cherries": "High",
    "parmesan cheese": 0,
    "bacon": 0,
    "marble cake muffin": "High",
    "lemon muffin": "High",
    "gf bread": 0,
    "crisps half": 0,
    "white rice": 0,
    "red cabbage": 0,
    "potatoes dinner": 0,
    "white pasta half": "High",
    "green pepper": 0,
    "sable": "High",
    "zockerwäffelchen": "High",
    "marble cake": "High"
                  }

    fodmap_lst = []

    for i, r in df.iloc[:200].iterrows():
        sum = 0
        print(i)
        r = r.str.lower()
        r.fillna("none", inplace=True)

        for k in r:
            try:
                sum = sum + 1 if fodmap_dic[k.lower()] == "High" else sum

            except:
                continue

        fodmap_lst.append(sum)
        df["fodmap_qty"] = pd.Series(fodmap_lst)


def get_lagged_fat(df):

    fat_dic = {"egg":5, "eggs":10, "eggsx2":10, "eggs x2": 10, "eggsx3":15, "crisps":17.5, "bacon":10.5, "chorizo butter":5, "butter chorizo":5,
               "chorizo":2,"salmon":30, "tuna mayo sriracha":7.5, "mayo sriracha":7.5, "mayo": 5, "nutella": 6, "yoghurt": 2,
               "madeleinesx2": 10, "gf madeleine": 5, "gf speculoos": 4, "gf bread": 1.5, "turkey": 7, "dark chocolate": 4,
               "thuringer": 20, "thüringer": 20, "sardines": 11, "olives": 10, "brochettes": 14, "oats": 3.5, "potatoes-dinner": 4,
               "crisps half": 8.75, "ham": 2.5, "huit": 8, "steak": 25, "gf chicken fingers": 11,
               "wainzoosis": 20, "comte": 3, "cake": 11, "dinde roulade": 10, "Walkers Shortbread": 7, "maiffelcher":20,
               "avocado": 8, "salmon half": 12, "peanuts":10, "gf chicken fingers half": 5.5, "chicken": 10,
               "cordon bleu": 15, "Mäiffelcher": 15, "Chorizo": 14, "sable": 4.5}

    healthy_fat_dic = {"sardines": 11 ,"turkey": 7, "dinde roulade": 10, "salmon": 30, "olives": 10, "egg": 5, "eggsx2": 10,
                       "avocado": 9.5, "gf chicken fingers": 9, "salmon half": 12, "gf bread": 1.5, "peanuts": 10,
                       "gf chicken fingers half": 4, "chicken": 8}

    fat_lst = []
    healthy_fat_lst = []

    for i, r in df.iloc[:200].iterrows():
        healthy_sum = 0
        sum = 0
        print(i)
        r = r.str.lower()
        r.fillna("none", inplace=True)
        extra = 14 if "potatoes" in r.values else 0

        for k in r:
            try:
                item_extra = 10 if "olive oil" in k else 0
                sum += (fat_dic[k] + item_extra)
                healthy_sum += (healthy_fat_dic[k]+item_extra)

            except:
                continue

        fat_lst.append(sum + extra)
        healthy_fat_lst.append(healthy_sum + extra)

    df["fat_qty"] = pd.Series(fat_lst)
    df["healthy_fat_pct"] = pd.Series(healthy_fat_lst)/pd.Series(fat_lst)
    fat_1l, fat_1l_pct = df["fat_qty"].shift(-1).rename("qty1l"), df["healthy_fat_pct"].shift(-1).rename("pct1l")
    #fat_1l, fat_2l = df["fat_qty"].shift(-1).rename("qty1l"), df["fat_qty"].shift(-2).rename("qty2l")
    #calories_1l, calories_2l = df["Cal"].shift(-1), df["Cal"].shift(-2)
    #fat_df = pd.concat([df["Consistency"], fat_1l, fat_1l / (calories_1l/100), fat_2l, fat_2l/(calories_2l/100)], axis=1)

    fat_df = pd.concat([df["Consistency"], fat_1l, fat_1l_pct], axis=1)

    return fat_df

def get_lagged_fiber(df):

    soluble_ratio_l1 = (df['Soluble'].shift(-1) / df["Total Fiber"].shift(-1)).rename("soluble_l1")
    soluble_ratio_l2 = (df['Soluble'].shift(-2) / df["Total Fiber"].shift(-2)).rename("soluble_l2")

    fiber_l1 = (df['Total Fiber'].shift(-1)).rename("fiber_l1")
    fiber_l2 = (df['Total Fiber'].shift(-2)).rename("fiber_l2")

    sdf = pd.concat([df["Consistency"], df["Emptying"], soluble_ratio_l1, soluble_ratio_l2, fiber_l1, fiber_l2], axis = 1)
    sample = sdf[sdf["Consistency"].str.lower() != "non-floating stool - shaped"]
    sample_normal = sdf[sdf["Consistency"].str.lower() == "non-floating stool - shaped"]


    return {"non-normal soluble": sample.median(),
            "normal soluble": sample_normal.median()}

def get_lagged_fructose(df):

    fructose_dic = {
        "broccoli": 1,
        "quinoa": 0.1,
        "raspberry": 2.05,
        "potatoes": 0.23,
        "avocado": 0.2,
        "blueberry": 5,
        "apple": 5.9,
        "tomatoes": 1.4,
        "gf gnocchi": 5.9,
        "pears": 6.4,
        "almonds": 0.4,
        "banana": 3,
        "oats": 0,
        "red pepper": 2.6,
        "white bread": 0,
        "onion": 3.3,
        "garlic": 0,
        "basil": 0,
        "pine nuts": 0,
        "white pasta": 0,
        "kiwi": 4.4,
        "carrots": 2.5,
        "veggie soup": 0,
        "tomatoe soup": 2.5,
        "peas": 0.4,
        "dark chocolate": 0,
        "whole grain bread": 0,
        "sweet potato": 0.7,
        "chestnut mushroom": 0.2,
        "green beans": 1,
        "endive": 0,
        "peanuts": 0.1,
        "baked beans": 4,
        "courgette": 1,
        "strawberry": 3.8,
        "psyllium": 0,
        "raisins": 9.7,
        "olives": 0,
        "apricot": 1.3,
        "batavia": 0,
        "crisps": 0,
        "cherries": 4.6,
        "bacon": 0,
        "marble cake muffin": 0,
        "gf Bread": 0.6,
        "crisps half": 0,
        "white Rice": 0,
        "red cabbage": 1.4,
        "nutella": 0.63/5,
        "jam": 23/10,
        "sorbet": 20,
        "gf speculoos": 25/4,
        "Boxemännchen": 39/2 * 1.25,
        "Boxemännchen_half": 39/2 * 1.25/2

    }

    fructose_lst = []

    for i, r in df.iloc[:200].iterrows():
        sum = 0
        print(i)
        r = r.str.lower()
        r.fillna("none", inplace=True)

        for k in r:
            try:
                sum += fructose_dic[k]

            except:
                continue

        fructose_lst.append(sum)

    df["fructose_qty"] = pd.Series(fructose_lst)
    fruc_1l, fruc_2l = df["fructose_qty"].shift(-1).rename("fr_qty1l"), df["fructose_qty"].shift(-2).rename("fr_qty2l")
    fat_df = pd.concat([df["Consistency"], fruc_1l, fruc_2l], axis=1)

    return fat_df

def get_lagged_psyllium(df):

    psy_lst = []
    for i, r in df.iloc[:200].iterrows():
        r = list(r.str.lower().values)
        c = r.count("gf bread")
        psy_lst.append(c)

    df["psyllium"] = pd.Series(psy_lst)
    psy_1l, psy_2l = df["psyllium"].shift(-1).rename("psy_qty1l"), df["psyllium"].shift(-2).rename("psy_qty2l")
    psy_df = pd.concat([df["Consistency"], psy_1l, psy_2l], axis = 1)

    return psy_df

def get_lactose(df):

    lactose_foods = ["comte", "camembert", "schnitzel dauphinois",
                     "dauphinois", "yoghurt", "pommerloch salad", "emmentaler", "trio pasta", "marble cake muffin",
                     "marble cake", "huit", "maiffelcher", "raspberry muffin", "streusel", "cake", "marmorkuch", "petit beurre",
                     "zockerwafel", "yoghurt", "korinthkuch", "korinthkuchx2", "almond cake", "marble cake muffinx2",
                     "granule", "parmesan", "risotto", "petit beurres", "rhubarb tart", "herb butter", "chese ham herb butter",
                     "oyster risotto", "crème dressing", "muffin", "speculoos", "cookies", "compte", "schnitzel dauphinois",
                     "parmesan cheese", "quiche", "croque m", "cheese ham", "walkers shortbread", "butter chorizo comte",
                     "ceasar salad", "croissant", "marble muffin", "mousse / vanilla", "gouda", "emmental", "pommerloch salad",
                     "emmentaler pickles", "camembert sausage", "gf bread 0.5 + comte", "marmorkuch", "camembert compte", "chorizo camembert",
                     "croque monsieur", "nikki marquisettes", "dinde roulade", "herb butter", "mäiffelcher",
                     "cordon bleu"]

    lactose_lst = []

    for i, r in df.iloc[:200].iterrows():
        c = 0
        r = list(r.str.lower().values)
        for l in lactose_foods:
            c = c+1 if l in r else c
        #c = 1 if any([l for l in lactose_foods if l in r]) is True else 0

        lactose_lst.append(c)

    df["lactose"] = pd.Series(lactose_lst)
    lac_1l, lac_2l = df["lactose"].shift(-1).rename("lac_qty1l"), df["lactose"].shift(-2).rename("lac_qty2l")
    lac_df = pd.concat([df["Consistency"], lac_1l, lac_2l], axis = 1)

    return lac_df

def get_gluten(df):

    gluten_lst = ["White Bread", "Whole Grain Bread", "Zockerwäffelchen x2", "Zockerwäffelcher x2", "Cookies",
                  "Sable", "Almond Cookie", "Marble Cake", "White Pasta", "White Pasta Half", "Mäiffelcher"]
    gluten_count_lst = []

    for i, r in df.iloc[:200].iterrows():
        c = 0
        r = list(r.str.lower().values)

        for l in gluten_lst:
            c = c+1 if l.lower() in r else c
        #c = 1 if any([l for l in lactose_foods if l in r]) is True else 0

        gluten_count_lst.append(c)
        df["gluten"] = pd.Series(gluten_count_lst)

    return gluten_count_lst

def get_irregularities(df):

    import random
    _df = pd.DataFrame()

    flag = False
    for i,r in df.iterrows():
        try:
            if "murky" in r["Other"].lower():
                flag = True
                idx = i
                ref_id = random.randint(0, 1000)
                r["ref_id"] = ref_id
                #_df = pd.concat([_df, r], axis = 0)
                _df = _df.append(r)
                continue

            if flag is True and idx + 2 >= i > idx:

                #_df = pd.concat([_df, r], axis=0)
                r["ref_id"] = ref_id
                _df = _df.append(r)

            else:
                flag = False

        except:
            continue

    return _df

def get_common_denominators(df):

    dfi = pd.DataFrame()
    irreg = get_irregularities(df)
    for x in irreg["ref_id"].unique():
        selection = irreg[irreg["ref_id"] == x].iloc[-1]
        dfi = dfi.append(selection)

    return dfi



if __name__ == "__main__":

    #get_lagged_fat(df)
    #get_lagged_psyllium(df)
    #lac_df = get_lactose(df)
    #f = get_fodmap_score(df)
    #_df = get_irregularities(df)
    #fdf = get_lagged_fat(df)
    #_df = get_common_denominators(df)
    #gdf = get_gluten(df)
    df["fiber_ratio"] = df["insoluble"] / df["Soluble"]
    di = DietInspector(df)
    analyt = di.analytics(df)
    sample = di.rolling_exposure(n_days=2, target = None, master_df=df, target_df=df)
    #b = di.abnormality_df(col = "fiber ratio", lag = 1)
