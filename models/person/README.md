# Person models

This folder contains all the `person` models that are used in the Covid-19 health management system.

The `person` model creates people samples with the necessary people attributes (e.g. age, commute option, zipcode, etc.). Unlike other models, the samples do not change over time. This is because `person` models have no concept of time: the populations are immutable over time (their only time attribute is their return of campus). However, you are encouraged to implement mutable populations over time.

Here, we share our full fledged person model that creates such samples.

Note that the `number of people samples` is different from the `size of the population` you want to model. It is recommended to have large multiples of samples compared to the population (e.g. `no. of samples = 20 x population size`).

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
        {
            "mit_person_type": "Admin staff",
            "home_zip": "01845",
            "commuting_type": "mbta",
            "age": 14,
            "building_and_core_access": [
                [
                    {
                        "building": "18"
                    }
                ]
            ],
            "first_day_returning": "2020-06-28"
        },
        {
            "mit_person_type": "Research staff",
            "home_zip": "01451",
            "commuting_type": "drive own vehicle",
            "age": 18,
            "building_and_core_access": [
                [
                    {
                        "building": "W92"
                    }
                ]
            ],
            "first_day_returning": "2020-07-22"
        },
        ...
        ]
}


```
