import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Style
plt.style.use('ggplot')

print("IPL DATA ANALYSIS PROJECT (MASTER VERSION)")
print("------------------------------------------")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("ipl_balls.csv")

print("\nColumns:")
print(df.columns)

print("\nSample Data:")
print(df.head())

# =========================
# DATA CLEANING
# =========================
df.dropna(subset=["batter", "bowler"], inplace=True)
df.fillna(0, inplace=True)

print("\nData cleaned successfully")

# =========================
# BASIC ANALYSIS
# =========================
total_runs = df["runs_batter"].sum()
total_wickets = df["wicket_taken"].sum()

print("\nTotal Runs:", total_runs)
print("Total Wickets:", total_wickets)

# =========================
# FUNCTION (Reusable)
# =========================
def show_top_players(data, column):
    top = data.groupby(column)["runs_batter"].sum().sort_values(ascending=False).head(5)
    print(f"\nTop {column}:")
    print(top)

show_top_players(df, "batter")

# =========================
# TOP BATSMEN
# =========================
top_batsman = df.groupby("batter")["runs_batter"].sum().sort_values(ascending=False).head(10)

colors = ['red','blue','green','orange','purple','cyan','magenta','yellow','brown','pink']

plt.figure()
top_batsman.plot(kind="bar", color=colors)
plt.title("Top 10 Batsmen")
plt.xlabel("Players")
plt.ylabel("Runs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_batsmen.png")
plt.show()

# =========================
# TOP BOWLERS
# =========================
top_bowler = df.groupby("bowler")["wicket_taken"].sum().sort_values(ascending=False).head(10)

plt.figure()
top_bowler.plot(kind="bar", color='green')
plt.title("Top 10 Bowlers")
plt.xlabel("Players")
plt.ylabel("Wickets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_bowlers.png")
plt.show()

# =========================
# RUNS PER OVER
# =========================
runs_over = df.groupby("over")["runs_batter"].sum()

plt.figure()
runs_over.plot(color='purple', linewidth=2)
plt.title("Runs per Over")
plt.xlabel("Over")
plt.ylabel("Runs")
plt.grid()
plt.savefig("runs_over.png")
plt.show()

# =========================
# PIE CHART
# =========================
dismissal_clean = df[df["dismissal_kind"] != 0]["dismissal_kind"]

# Count top dismissals
dismissal = dismissal_clean.value_counts().head(5)

plt.figure(figsize=(7,7))

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']

plt.pie(
    dismissal,
    labels=dismissal.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors
)

plt.title("Top 5 Dismissal Types", fontsize=14)
plt.axis('equal')

plt.show()


plt.title("Top 5 Dismissal Types", fontsize=14)
plt.axis('equal')  # makes circle perfect

plt.show()
# =========================
# REGRESSION
# =========================
runs_over_df = df.groupby("over")["runs_batter"].sum().reset_index()

X = runs_over_df[["over"]]
y = runs_over_df["runs_batter"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure()
plt.scatter(X, y, color='blue', label="Actual")
plt.plot(X, y_pred, color='red', linewidth=3, label="Regression Line")
plt.title("Regression: Over vs Runs")
plt.xlabel("Over")
plt.ylabel("Runs")
plt.legend()
plt.grid()
plt.savefig("regression.png")
plt.show()

score = model.score(X, y)
print("\nRegression Accuracy Score:", score)

# =========================
# STRIKE RATE
# =========================
balls_faced = df.groupby("batter").size()
runs_scored = df.groupby("batter")["runs_batter"].sum()

strike_rate = (runs_scored / balls_faced) * 100
top_sr = strike_rate.sort_values(ascending=False).head(10)

print("\nTop Strike Rate Players:")
print(top_sr)

plt.figure()
top_sr.plot(kind="bar", color='magenta')
plt.title("Top Strike Rate Players")
plt.xlabel("Players")
plt.ylabel("Strike Rate")
plt.xticks(rotation=45)
plt.savefig("strike_rate.png")
plt.show()

# =========================
# TEAM ANALYSIS
# =========================
team_runs = df.groupby("batting_team")["runs_batter"].sum().sort_values(ascending=False)

plt.figure()
team_runs.plot(kind="bar", color='teal')
plt.title("Total Runs by Team")
plt.xlabel("Team")
plt.ylabel("Runs")
plt.xticks(rotation=45)
plt.savefig("team_runs.png")
plt.show()

# =========================
# CORRELATION
# =========================
correlation = df[["runs_batter", "runs_extras", "wicket_taken"]].corr()
print("\nCorrelation Matrix:\n", correlation)

# =========================
# HISTOGRAM
# =========================
plt.figure()
df["runs_batter"].plot(kind="hist", bins=10, color='skyblue')
plt.title("Distribution of Runs per Ball")
plt.xlabel("Runs")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
plt.show()

# =========================
# BOXPLOT
# =========================
plt.figure()
df.boxplot(column="runs_batter")
plt.title("Box Plot of Runs")
plt.savefig("boxplot.png")
plt.show()

# =========================
# BEST OVER
# =========================
best_over = runs_over.idxmax()
print("\nBest Scoring Over:", best_over)

# =========================
# SUMMARY
# =========================
print("\nSUMMARY:")
print("Total Matches:", df["match_id"].nunique())
print("Total Players:", df["batter"].nunique())

# =========================
# INSIGHTS
# =========================
print("\nINSIGHTS:")
print("- Top batsmen dominate scoring")
print("- Death overs have higher scoring")
print("- Some players have very high strike rates")
print("- Team performance varies significantly")

print("\nProject Completed Successfully!")
