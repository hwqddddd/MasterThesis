from os import path
import io
from json import load, dump

package_path = path.join('data/Amazon_data/package_data.json')
with open(package_path, newline='') as in_file:
    dataPackage = load(in_file)

routes_path = path.join('data/Amazon_data/route_data.json')
with open(routes_path, newline='') as in_file:
    dataRoute = load(in_file)

travel_path = path.join('data/Amazon_data/travel_times.json')
with open(travel_path, newline='') as in_file2:
    dataTravel = load(in_file2)


i = 0

for route in dataRoute:
    if i < 15:    
        n_customer = len(dataRoute[route].get('stops'))
        # if n_customer > 100 and n_customer < 200:
        if n_customer > 200:
            json_data = {}
            json_data['route_ID'] = route
            json_data['vehicle_capacity'] = dataRoute[route]['executor_capacity_cm3']
            json_data['Number_of_customers'] = len(dataRoute[route].get('stops'))
            start = dataRoute[route]['departure_time_utc']
            start = int(start[0:2])*3600 + int(start[3:5])*60 + int(start[6:])

            j = 0
            for stop in dataRoute[route].get('stops').keys():            
                demand = 0
                service_time = 0
                ready_time = 86400
                due_time = 0
                for package in dataPackage[route][stop].keys():
                    dimension = dataPackage[route][stop][package]['dimensions']
                    demand += dimension['depth_cm'] * dimension['height_cm'] * dimension['width_cm']
                    service_time += dataPackage[route][stop][package]['planned_service_time_seconds']
                    ready = dataPackage[route][stop][package]['time_window']['start_time_utc']
                    if ready == ready:
                        ready = int(ready[11:13])*3600 + int(ready[14:16])*60 + int(ready[17:])

                        ready =  ready - start

                        if ready < ready_time:
                            ready_time = ready


                    end = dataPackage[route][stop][package]['time_window']['end_time_utc']
                    if end == end:

                        end = int(end[11:13])*3600 + int(end[14:16])*60 + int(end[17:])

                        end = end -start
                        if end > due_time:
                            due_time = end
                        elif end == 0:
                            due_time = 86400
    

                if ready_time == 86400:
                    ready_time = 0
                    due_time = 86400
                json_data[f'customer_{j}'] = {
                            'coordinates': {
                                'x': dataRoute[route]['stops'][stop]['lat'],
                                'y': dataRoute[route]['stops'][stop]['lng'],
                            },
                            'demand': demand,
                            'ready_time': ready_time,
                            'due_time': due_time,
                            'service_time': service_time,
                        }
                j += 1
            json_data['distance_matrix'] = [[dataTravel[route][stop1][stop2] for stop1 in dataTravel[route][stop].keys()] for
                                                    stop2 in dataTravel[route][stop].keys()]

            json_data['instance_name'] = 'A30' + str(i)
            json_file_name = f"{json_data['instance_name']}.json"
            json_file = path.join('data/Amazon_data/new_new_json', json_file_name)
            with io.open(json_file, 'wt', newline='') as file_object:
                dump(json_data, file_object, sort_keys=True, indent=4, separators=(',', ': '))
            i += 1
