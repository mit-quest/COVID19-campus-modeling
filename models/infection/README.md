# Infection models

This folder contains all the `infection` models that are used in the Covid-19 health management system.

This is the model which takes in the other models and simulates whether and when each person sample gets infected. Normally, you want to have multiple `infection` models which model different types of infections at the same time: a commuting model to model infections while commuting, a model for modeling in-building infection as people share spaces in time and space, a model for local activities infection, etc.

Currently, the `infection_demo` model is a placeholder model.

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
            false,
            false,
            false
        ],
        [
            false,
            false,
            false
        ],
        [
            false,
            true,
            false
        ],
        [
            false,
            true,
            false
        ],
        [
            false,
            true,
            false
        ]
    ]
}
```
