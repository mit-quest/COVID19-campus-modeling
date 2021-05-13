# Action models

This folder contains all the `trajectory` models that are used in the Covid-19 health management system.

The `trajectory` model takes in the `person` model samples and creates daily trajectories of people bumbling through buildings, including splitting overnight trajectories into different days. The code also supports local activities around and outside campus e.g. going to lunch.

Here, we share our full fledged `trajectory` model that creates such samples. Empty samples means that this person did not go to work this day.

## Running a model
Refer to instructions in the root folder

## Creating new models
Refer to instructions in the root folder

## Example output JSON
```
{
    "dates": [
        "2020-08-01",
        "2020-08-02",
        "2020-08-03"
    ],
    "samples": [
        [
            [],
            [],
            []
        ],
        [
            [],
            [],
            []
        ],
        [
            [],
            [],
            []
        ],
        [
            [],
            [],
            [
                {
                    "stay_type": "commute",
                    "commute_type": "drive own vehicle",
                    "start_zip_code": "02126",
                    "arrival_time": "14:05"
                },
                {
                    "stay_type": "on_campus_inside",
                    "building": "outside",
                    "start_time": "14:05",
                    "end_time": "15:00"
                },
                {
                    "stay_type": "local_activity",
                    "location": null,
                    "start_time": "15:00",
                    "end_time": "15:33"
                },
                {
                    "stay_type": "on_campus_inside",
                    "building": "outside",
                    "start_time": "15:33",
                    "end_time": "19:51"
                },
                {
                    "stay_type": "commute",
                    "commute_type": "drive own vehicle",
                    "end_zip_code": "02126",
                    "departure_time": "19:51"
                }
            ]
        ],
        [
            [],
            [],
            []
        ]
    ],
    "total_population_size": 5069
}
```
