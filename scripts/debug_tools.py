"""Small typed helpers used to verify the project's development tools."""


def normalize_score(score: float, minimum: float, maximum: float) -> float:
    """Scale a score to the range from zero to one."""
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    return (score - minimum) / (maximum - minimum)


def main() -> None:
    """Print one sample normalization result."""
    print(f"Normalized score: {normalize_score(75.0, 0.0, 100.0):.2f}")


if __name__ == "__main__":
    main()
