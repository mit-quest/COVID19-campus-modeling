# Prevalence models

This folder contains all the `prevalence` models that are used in the Covid-19 health management system.

The role of the `prevalence` model is to provide local (zipcode level) disease prevalences. This prevalence will be used to simulate infection at the local level as people are transiting through different zipcodes. Currently, the `prevalence_demo` model is a placeholder model that just produces randomly generated prevalences for all zipcodes under consideration.

This is where you can either implement your own SEIR model, or query APIs of prevalences (current and forecasted).

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
            "0": {
                "01451": {
                    "prevalence_uncontained_infections": 0.003031552592532352
                },
                "01453": {
                    "prevalence_uncontained_infections": 0.002646071577263719
                },
                "01460": {
                    "prevalence_uncontained_infections": 0.0030744209036917573
                },
                "01545": {
                    "prevalence_uncontained_infections": 0.003189960562639481
                },
                "01701": {
                    "prevalence_uncontained_infections": 0.002507849342339896
                },
                "01702": {
                    "prevalence_uncontained_infections": 0.0027675777730497824
                },
                "01719": {
                    "prevalence_uncontained_infections": 0.0028671336254055104
                },
                "01720": {
                    "prevalence_uncontained_infections": 0.0024020091294390407
                },
                "01721": {
                    "prevalence_uncontained_infections": 0.002806315666074312
                },
                "01730": {
                    "prevalence_uncontained_infections": 0.002393419403090526
                },
                ...
                ]
            ...
        }
    ...
    ]
}
```
