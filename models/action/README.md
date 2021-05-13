# Action models

This folder contains all the `action` models that are used in the Covid-19 health management system.

This is the model where you can implement different policy actions. For example, a schedule (ramp) of reopening or closing buildings over time. Currently, the `action_demo` model is a placeholder model that takes a percentage reopening and creates a (randomly sampled) start and end date of reopening for each building.


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
            {
                "1": 0.0,
                "10": 0.0,
                "11": 0.0,
                "12": 0.13,
                "13": 0.0,
                "14": 0.0,
                "16": 0.0,
                "17": 0.44,
                "18": 0.0,
                "2": 0.0,
                ...
            },
            ...]
}
```
