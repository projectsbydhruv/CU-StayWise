from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
from livability_model import (LivabilityModel, COMPLAINT_CATEGORIES)

app = Flask(__name__)

# Load data on startup
print("Loading data...")
df = pd.read_csv("final_integrated.csv")
model = LivabilityModel(df)


@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    """
    Livability Score homepage.

    - GET: show empty page + form
    - POST: use current livability search logic and send results to template
    """
    building = ""
    street = ""
    zipcode = ""
    results = None
    error = None

    if request.method == "POST":
        building = request.form.get("building", "").strip()
        street = request.form.get("street", "").strip()
        zipcode = request.form.get("zipcode", "").strip()

        if not any([building, street, zipcode]):
            error = "Please enter at least one of: building number, street name, or ZIP code."
        else:
            matches = model.search_address(
                building=building,
                street=street,
                zipcode=zipcode,
                max_results=5
            )
            if not matches:
                error = (
                    "We couldn't find a relevant building in our dataset for that input. "
                    "Try adjusting the street spelling or ZIP code."
                )
            else:
                results = matches

    return render_template(
        "home.html",
        building=building,
        street=street,
        zipcode=zipcode,
        results=results,
        error=error
    )


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    """Summary / overview page with Ghost Landlord detector."""
    df_all = model.df.copy()

    # Find complaint category column
    candidate_cols = [
        "Complaint_Category", "complaint_category", "Category",
        "Complaint_Type", "complaint_type",
    ]
    cat_col = None
    for c in candidate_cols:
        if c in df_all.columns:
            cat_col = c
            break
    if cat_col is None:
        raise RuntimeError(f"Could not find complaint category column. Available: {list(df_all.columns)}")

    # High-level stats
    total_complaints = len(df_all)
    total_buildings = df_all["bbl"].nunique()
    total_zipcodes = df_all["incident_zip"].nunique()

    # Top complaint categories
    cat_counts = df_all[cat_col].value_counts().reset_index(name="count")
    cat_counts.columns = ["category", "count"]
    top_categories = [{"category": row["category"], "count": int(row["count"])}
                      for _, row in cat_counts.iterrows()]
    top_chart = top_categories[:5]
    top_chart_labels = [c["category"] for c in top_chart]
    top_chart_counts = [c["count"] for c in top_chart]

    # ZIP breakdown
    zip_group = df_all.groupby(["incident_zip", cat_col]).size().reset_index(name="count")
    zip_breakdown = []
    for zip_code, group in zip_group.groupby("incident_zip"):
        group_sorted = group.sort_values("count", ascending=False)
        most = group_sorted.iloc[0]
        least = group_sorted.iloc[-1]
        zip_breakdown.append({
            "zip": int(zip_code),
            "most_cat": most[cat_col],
            "most_cnt": int(most["count"]),
            "least_cat": least[cat_col],
            "least_cnt": int(least["count"]),
        })
    zip_breakdown = sorted(zip_breakdown, key=lambda z: z["zip"])

    # Best buildings to live in - HIGHEST LIVABILITY SCORES
    print("Calculating livability scores for all buildings...")

    # Get BBLs that actually have scores (from the pre-computed building_scores_all)
    # instead of iterating over ALL BBLs in the raw data
    if hasattr(model, 'building_scores_all') and model.building_scores_all is not None:
        valid_bbls = model.building_scores_all['bbl'].unique()
    else:
        print("WARNING: No building_scores_all found, using df BBLs")
        valid_bbls = df_all['bbl'].unique()

    all_buildings_with_scores = []
    for bbl in valid_bbls:
        livability_info = model.get_building_livability(bbl)
        if livability_info:
            score = livability_info.get('livability_score')
            # Only include buildings with valid numeric scores
            if score is not None and isinstance(score, (int, float)):
                all_buildings_with_scores.append(livability_info)

    print(f"DEBUG: Total buildings with valid scores: {len(all_buildings_with_scores)}")

    # Sort by livability score (highest first) and take top 10
    all_buildings_with_scores.sort(key=lambda x: x['livability_score'], reverse=True)
    best_buildings = [
        {
            "address": b['address'],
            "zip": b['zipcode'],
            "count": b['complaint_count'],
            "livability_score": b['livability_score']
        }
        for b in all_buildings_with_scores[:10]
    ]

    # Worst buildings - LOWEST LIVABILITY SCORES
    all_buildings_with_scores.sort(key=lambda x: x['livability_score'])
    worst_buildings = [
        {
            "address": b['address'],
            "zip": b['zipcode'],
            "count": b['complaint_count'],
            "livability_score": b['livability_score']
        }
        for b in all_buildings_with_scores[:10]
    ]

    print(f"DEBUG: Best buildings count: {len(best_buildings)}")
    print(f"DEBUG: Worst buildings count: {len(worst_buildings)}")
    if len(worst_buildings) > 0:
        print(f"DEBUG: Sample worst building: {worst_buildings[0]}")

    # GHOST LANDLORD DETECTOR
    building_query = ""
    res_error = None
    res_building = None

    if request.method == "POST":
        building_query = request.form.get("building_query", "").strip()

        if not building_query:
            res_error = "Please enter an address, ZIP code, or BBL."
        else:
            subset = None
            q = building_query.upper()

            # Try BBL (10 digits)
            if q.isdigit() and len(q) == 10 and "bbl" in df_all.columns:
                subset = df_all[df_all["bbl"].astype(str) == q]

            # Try ZIP (5 digits)
            elif q.isdigit() and len(q) == 5:
                subset = df_all[df_all["incident_zip"].astype(str) == q]

            # Try partial address match
            else:
                if "Incident_Address" in df_all.columns:
                    subset = df_all[
                        df_all["Incident_Address"].astype(str).str.upper().str.contains(q, na=False)
                    ]

            if subset is None or subset.empty:
                res_error = "No complaints found for that input. Try a different address or ZIP code."
            else:
                # Find date columns
                created_col = None
                closed_col = None

                for col in ['Created_Date', 'Created_DateTime', 'created_date']:
                    if col in subset.columns:
                        created_col = col
                        break

                for col in ['Closed_Date', 'Closed_DateTime', 'closed_date']:
                    if col in subset.columns:
                        closed_col = col
                        break

                if created_col is None or closed_col is None:
                    res_error = "Date columns not found in dataset. Cannot calculate resolution time."
                else:
                    # Calculate resolution times
                    rt_df = subset[[created_col, closed_col]].copy()
                    rt_df["created"] = pd.to_datetime(rt_df[created_col], errors="coerce")
                    rt_df["closed"] = pd.to_datetime(rt_df[closed_col], errors="coerce")

                    mask = rt_df["created"].notna() & rt_df["closed"].notna()
                    rt_series = ((rt_df.loc[mask, "closed"] - rt_df.loc[mask, "created"]).dt.total_seconds() / 86400.0)
                    rt_series = rt_series[rt_series >= 0]

                    if rt_series.empty:
                        res_error = "Not enough closed complaint data to estimate resolution time for this location."
                    else:
                        avg_days = round(float(rt_series.mean()), 1)
                        n = int(len(rt_series))

                        if avg_days < 3:
                            label = "Speedy"
                        elif avg_days <= 7:
                            label = "Meh"
                        elif avg_days <= 14:
                            label = "Slow"
                        else:
                            label = "Ghosted"

                        addr = None
                        zipc = None
                        if "Incident_Address" in subset.columns and subset["Incident_Address"].notna().any():
                            addr = str(subset.iloc[0]["Incident_Address"])
                        if "incident_zip" in subset.columns and subset["incident_zip"].notna().any():
                            zipc = int(subset.iloc[0]["incident_zip"])

                        res_building = {
                            "query": building_query,
                            "address": addr,
                            "zipcode": zipc,
                            "avg_days": avg_days,
                            "label": label,
                            "num_complaints": n,
                        }

    return render_template(
        "dashboard.html",
        total_complaints=total_complaints,
        total_buildings=total_buildings,
        total_zipcodes=total_zipcodes,
        top_chart_labels=top_chart_labels,
        top_chart_counts=top_chart_counts,
        zip_breakdown=zip_breakdown,
        low_buildings=best_buildings,  # Now highest livability scores
        high_buildings=worst_buildings,  # Now lowest livability scores
        building_query=building_query,
        res_error=res_error,
        res_building=res_building,
    )


@app.route("/map")
def index():
    """Interactive map page"""
    # pass just the list of category names, which the JS expects
    return render_template("index.html", categories=list(COMPLAINT_CATEGORIES.keys()))


@app.route("/api/zip_scores", methods=["GET"])
def api_zip_scores():
    """API endpoint to get zipcode-level scores for selected categories."""
    categories = request.args.getlist("categories")

    if not categories:
        categories = list(COMPLAINT_CATEGORIES.keys())

    try:
        scores = model.calculate_zipcode_scores_by_category(selected_categories=categories)
        data = scores.to_dict(orient='records')
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route("/api/calculate_scores", methods=["POST"])
def api_calculate_scores():
    """API endpoint to calculate zipcode scores (POST version for the map)."""
    data = request.get_json() or {}
    categories = data.get("categories", list(COMPLAINT_CATEGORIES.keys()))

    try:
        scores = model.calculate_zipcode_scores_by_category(selected_categories=categories)
        zipcodes = scores.to_dict(orient='records')
        return jsonify({'success': True, 'zipcodes': zipcodes})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route("/api/building_scores", methods=["GET"])
def api_building_scores():
    """API endpoint to get building-level scores for a given ZIP and categories."""
    zipcode = request.args.get("zipcode", type=int)
    categories = request.args.getlist("categories")

    if not categories:
        categories = list(COMPLAINT_CATEGORIES.keys())

    try:
        building_scores = model.calculate_building_scores_by_category(
            selected_categories=categories
        )

        if zipcode:
            building_scores = building_scores[building_scores['incident_zip'] == zipcode]

        data = building_scores.to_dict(orient='records')
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route("/api/buildings/<int:zipcode>", methods=["GET"])
def api_buildings_by_zip(zipcode):
    """API endpoint to get building-level scores for a specific ZIP code."""
    # Handle both formats: JSON string or multiple params
    categories_param = request.args.get("categories")
    if categories_param:
        try:
            categories = json.loads(categories_param)
        except:
            categories = request.args.getlist("categories")
    else:
        categories = request.args.getlist("categories")

    if not categories:
        categories = list(COMPLAINT_CATEGORIES.keys())

    try:
        building_scores = model.calculate_building_scores_by_category(
            selected_categories=categories
        )

        building_scores = building_scores[building_scores['incident_zip'] == zipcode]
        buildings = building_scores.to_dict(orient='records')

        print(f"DEBUG: Buildings in zipcode {zipcode}: {len(buildings)}")
        return jsonify({'success': True, 'buildings': buildings})
    except Exception as e:
        print(f"ERROR in api_buildings_by_zip: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route("/api/all_buildings", methods=["GET"])
def api_all_buildings():
    """API endpoint to get all building-level scores (for the map view)."""
    # Handle both formats: JSON string or multiple params
    categories_param = request.args.get("categories")
    if categories_param:
        try:
            categories = json.loads(categories_param)
        except:
            categories = request.args.getlist("categories")
    else:
        categories = request.args.getlist("categories")

    if not categories:
        categories = list(COMPLAINT_CATEGORIES.keys())

    try:
        buildings = model.get_all_building_scores_by_category(selected_categories=categories)
        print(f"DEBUG: Number of buildings returned: {len(buildings)}")
        print(f"DEBUG: Categories requested: {categories}")
        if len(buildings) > 0:
            print(f"DEBUG: First building sample: {buildings[0]}")
        return jsonify({'success': True, 'buildings': buildings})
    except Exception as e:
        print(f"ERROR in api_all_buildings: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route("/api/building/<int:bbl>", methods=['GET'])
def get_building_details(bbl):
    """Get detailed information for a specific building (311 complaints only)."""
    try:
        building_complaints = model.df[model.df['bbl'] == bbl]

        if len(building_complaints) == 0:
            return jsonify({'success': False, 'error': 'Building not found'}), 404

        # Category breakdown
        breakdown = model.get_category_breakdown(bbl=bbl)

        # Recent complaints
        if 'year_created' in building_complaints.columns:
            recent = building_complaints.nlargest(10, 'year_created')[
                ['Complaint_Type', 'Category', 'year_created', 'Status', 'Descriptor']
            ].to_dict('records')
        else:
            cols = ['Complaint_Type', 'Category', 'Status', 'Descriptor']
            cols = [c for c in cols if c in building_complaints.columns]
            recent = building_complaints[cols].head(10).to_dict('records')

        # Building info (now only 311-complaint based)
        building_info = {
            'address': building_complaints.iloc[0]['Incident_Address'],
            'zipcode': int(building_complaints.iloc[0]['incident_zip']),
            'total_complaints': len(building_complaints),
            'latitude': float(building_complaints.iloc[0]['latitude']),
            'longitude': float(building_complaints.iloc[0]['longitude'])
        }

        return jsonify({
            'success': True,
            'building': building_info,
            'category_breakdown': breakdown,
            'recent_complaints': recent
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Basic stats for dashboard header."""
    try:
        total_complaints = int(len(model.df))
        total_buildings = int(model.df['bbl'].nunique())
        total_zips = int(model.df['incident_zip'].nunique())

        return jsonify({
            'success': True,
            'total_complaints': total_complaints,
            'total_buildings': total_buildings,
            'total_zips': total_zips
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route("/api/temporal", methods=["GET"])
def api_temporal():
    """Temporal trends for selected categories and/or ZIP."""
    categories = request.args.getlist("categories")
    zipcode = request.args.get("zipcode", type=int)

    try:
        selected_categories = categories if categories else None

        temporal_data = model.get_temporal_stats(
            selected_categories=selected_categories,
            zip_code=zipcode
        )

        return jsonify({
            'success': True,
            'temporal': temporal_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# JSON API version of the livability search
@app.route("/api/livability_search", methods=["GET", "POST"])
def livability_search():
    """JSON API version of the livability search (not used by home page now)."""
    if request.method == "POST":
        data = request.get_json() or {}
    else:
        data = request.args or {}

    building = data.get("building", "")
    street = data.get("street", "")
    zipcode = data.get("zipcode", "")

    try:
        results = model.search_address(
            building=building,
            street=street,
            zipcode=zipcode,
            max_results=5
        )
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)