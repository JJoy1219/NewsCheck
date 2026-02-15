"""
Test the live Hugging Face Gradio API with present-day articles (Feb 2026)
to verify model predictions across the political spectrum.
"""
import json
import urllib.request

SPACE_URL = "https://joeljoy1912-news-bias-classifier.hf.space"

articles = {
    "Article 1 - Progressive.org Op-Ed (Expected: Left)": (
        "On January 23, Vice President J.D. Vance launched the Trump Administration's "
        "new plan to promote families and human flourishing. But rather than being, as he "
        "claimed, pro-life, these rules will threaten the lives of people around the world, "
        "especially women and people who don't fit into the administration's narrow, "
        "unscientific categories of gender. The first of the new restrictions on foreign aid "
        "announced by Vance extends the existing Mexico City Policy prohibitions on abortion "
        "funding to encompass not just global health assistance but all non-military foreign "
        "assistance including U.S.-based nonprofits and government-to-government funding. "
        "Known to critics as the Global Gag Rule, recipients receiving funding from the U.S. "
        "government are prohibited even from tapping other donors to provide information or "
        "education regarding women's health. This expansion will severely limit access to "
        "abortion and the full range of sexual and reproductive health care, even in "
        "humanitarian emergencies. Decades of social science and health research show that "
        "curtailing access to safe abortions does not reduce abortions; it just makes them "
        "more dangerous. A 2017 World Health Organization report found that unsafe abortions "
        "already account for as much as 13.2 percent of maternal mortality worldwide. The "
        "Trump Administration's new rules will increase that needless suffering and death. "
        "The administration's grandstanding on abortion discounts the unimaginable suffering "
        "that we have seen in women who have died from unsafe abortions. The Global Gag Rule "
        "also undermines crucial health systems, forcing physicians and health workers to "
        "violate their professional oaths by staying silent when they see risks or to delay "
        "actions until it is too late to save their patients."
    ),

    "Article 2 - Fox News Opinion by Tim Graham (Expected: Right)": (
        "The Washington Post's announcement of 300 job cuts has drawn criticism from public "
        "media outlets NPR and PBS, who view the decision as a retreat from aggressive "
        "investigative journalism. Former Washington Post editor Marty Baron appeared on PBS "
        "to condemn owner Jeff Bezos's strategy. NPR and PBS journalists expect endless "
        "millions in subsidies to pursue agenda-driven reporting. They claim Bezos betrayed "
        "the paper's motto Democracy Dies in Darkness. But the critics overlook the Post's "
        "selective accountability. The paper won a Pulitzer for January 6 coverage but showed "
        "minimal interest in investigating Hunter Biden, suggesting double standards in which "
        "political figures receive scrutiny. The broadcast networks never tire of celebrating "
        "their liberal allies in the press while ignoring how one-sided their coverage truly "
        "is. Public media has no interest in being a fair arbiter of the news. They want to "
        "be advocates for progressive causes while pretending to serve all Americans. The bias "
        "is built into their DNA and no amount of taxpayer funding can change that fundamental "
        "reality."
    ),

    "Article 3 - Al Jazeera / CBO Fiscal Report (Expected: Center)": (
        "The Congressional Budget Office released a 10-year fiscal outlook showing "
        "deteriorating long-term deficits and rising national debt, driven primarily by "
        "increased spending on Social Security, Medicare, and debt service payments. The "
        "fiscal 2026 deficit is projected at approximately 5.8 percent of GDP, about where "
        "it was in 2025 when the deficit reached 1.775 trillion dollars. However, the "
        "deficit-to-GDP ratio will average 6.1 percent over the next decade, reaching 6.7 "
        "percent by 2036, far exceeding Treasury Secretary Scott Bessent's goal of roughly "
        "3 percent. According to the CBO, large deficits are unprecedented for a growing, "
        "peacetime economy, though there remains time for policymakers to correct course. "
        "The report factors in major developments including Republicans One Big Beautiful "
        "Bill Act, higher tariffs, and the Trump administration's immigration crackdown "
        "involving mass deportations. These changes increase the projected 2026 deficit by "
        "approximately 100 billion dollars, with total deficits from 2026-2035 running 1.4 "
        "trillion larger than previously estimated. Public debt is projected to rise from "
        "101 percent to 120 percent of GDP, exceeding historical highs. The CBO forecasts "
        "significantly lower growth than the Trump administration, projecting 2.2 percent "
        "real GDP growth for 2026, declining to approximately 1.8 percent for the remainder "
        "of the decade."
    ),

    "Article 4 - NPR News Roundup (Expected: Center)": (
        "The Trump administration signaled a softer approach to immigration enforcement in "
        "Minnesota, with border official Tom Homan announcing that 700 ICE agents would "
        "depart the state. A Democratic candidate achieved an unexpected victory in a special "
        "election for a Texas State Senate position, representing a potential setback for "
        "Republican leadership in Washington. The Department of Justice disclosed additional "
        "emails from financier Jeffrey Epstein's correspondence, featuring communications "
        "with prominent figures including Bill Gates and Elon Musk. Diplomatic talks between "
        "the United States and Iran experienced fluctuations throughout the week, with Arab "
        "nations urging the White House to maintain engagement. The Trump administration "
        "announced plans for a strategic mineral reserve and trade alliance designed to "
        "counteract China's dominance in rare earth metals and limit its negotiating leverage. "
        "Spain moved toward implementing age restrictions on social media access for minors, "
        "following Australia's comparable legislation."
    ),

    "Article 5 - Progressive News Service (Expected: Left)": (
        "Democrats in Congress are conditioning Department of Homeland Security funding "
        "approval on federal agent reform, demanding targeted enforcement, no masks, require "
        "ID, protect sensitive locations, stop racial profiling among ten specific "
        "requirements. NYC Mayor Zohran Mamdani and Rep. Alexandria Ocasio-Cortez are "
        "endorsing Gov. Kathy Hochul's reelection despite her centrist positions, dealing a "
        "significant setback to Lt. Gov. Antonio Delgado's progressive primary challenge. "
        "Trump allegedly offered to restore 16 billion in Hudson River tunnel funding if "
        "Senator Chuck Schumer supported renaming Washington Dulles Airport and New York "
        "Penn Station after Trump, a proposal Schumer rejected. Rep. Ted Lieu claims the "
        "released Epstein files contain highly disturbing allegations of Donald Trump and "
        "threatening to kill minors, urging press investigation. The U.S. has killed at "
        "least 119 people in strikes on suspected drug boats under Operation Southern Spear, "
        "with the Trump administration characterizing the campaign as an armed conflict "
        "against drug cartels. The UAW secured a tentative contract with Volkswagen including "
        "a 20 percent wage increase over four years and marked the first successful "
        "unionization of a foreign-owned, nonunion Southern automaker."
    ),
}


def predict(text):
    """Call the two-step Gradio API."""
    # Step 1: Submit
    req1 = urllib.request.Request(
        f"{SPACE_URL}/gradio_api/call/predict_bias",
        data=json.dumps({"data": [text]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req1, timeout=30) as resp1:
        event_id = json.loads(resp1.read().decode())["event_id"]

    # Step 2: Get result
    req2 = urllib.request.Request(
        f"{SPACE_URL}/gradio_api/call/predict_bias/{event_id}"
    )
    with urllib.request.urlopen(req2, timeout=60) as resp2:
        sse_text = resp2.read().decode()

    # Parse SSE stream
    lines = sse_text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("event: complete"):
            data_line = lines[i + 1]
            if data_line.startswith("data: "):
                output = json.loads(data_line[6:])
                label_data = output[0]
                confidences = {
                    c["label"]: c["confidence"] for c in label_data["confidences"]
                }
                return confidences
    return None


def score_to_label(score):
    if score <= -0.6:
        return "Left"
    elif score <= -0.2:
        return "Left-Leaning"
    elif score <= 0.2:
        return "Center"
    elif score <= 0.6:
        return "Right-Leaning"
    else:
        return "Right"


print("=" * 70)
print("LIVE MODEL TEST - Present-Day Articles (February 2026)")
print("=" * 70)
print()

results = []
for name, text in articles.items():
    try:
        probs = predict(text)
        if probs is None:
            print(f"{name}")
            print("  ERROR: No result from model")
            print()
            continue

        left = probs.get("Left", 0)
        center = probs.get("Center", 0)
        right = probs.get("Right", 0)
        score = (-1 * left) + (0 * center) + (1 * right)
        label = score_to_label(score)
        sign = "+" if score >= 0 else ""

        print(f"{name}")
        print(f"  Prediction:    {label} (score: {sign}{score:.3f})")
        print(f"  Probabilities: Left={left:.3f}  Center={center:.3f}  Right={right:.3f}")
        print()

        results.append({"name": name, "label": label, "score": score})

    except Exception as e:
        print(f"{name}")
        print(f"  ERROR: {e}")
        print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
correct = 0
total = len(results)
for r in results:
    name = r["name"]
    expected = name.split("Expected: ")[1].rstrip(")")
    predicted = r["label"]
    # Check if prediction is in the right direction
    match = False
    if expected == "Left" and predicted in ("Left", "Left-Leaning"):
        match = True
    elif expected == "Center" and predicted in ("Center", "Left-Leaning", "Right-Leaning"):
        match = True
    elif expected == "Right" and predicted in ("Right", "Right-Leaning"):
        match = True

    status = "CORRECT" if match else "WRONG"
    if match:
        correct += 1
    sign = "+" if r["score"] >= 0 else ""
    print(f"  {status:7s} | {predicted:14s} ({sign}{r['score']:.3f}) | {name.split(' - ')[1]}")

print()
print(f"Directional accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
