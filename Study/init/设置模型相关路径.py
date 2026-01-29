from ultralytics import settings

settings.update(
    {
        "runs_dir": "D:\\Work\\LLM\\GitHub\\ultralytics\\Study\\runs",
        "datasets_dir":"D:\\Work\\LLM\\GitHub\\ultralytics\\Study\\datasets",
        "weights_dir":"D:\\Work\\LLM\\GitHub\\ultralytics\\Study\\weights",
    }
)


# settings.reset()

print(settings)