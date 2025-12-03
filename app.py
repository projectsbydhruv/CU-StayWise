from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
from livability_model import (LivabilityModel, COMPLAINT_CATEGORIES)

app = Flask(__name__)

# Load data on startup
print("Loading data...")
df = pd.read_csv("final_integrated.csv")
model = LivabilityModel(df)

# Precompute data needed for the dashboard
dashboard_data = model.get_dashboard_summary()


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



@app.route("/overview")
def dashboard():
    """
    Quick Overview / summary statistics page.

    - Top 3 cards: total complaints, buildings, and ZIP codes
    - Best / worst ZIP, street, and building by livability score
    """

    # Header stats (simple counts from the main dataframe)
    total_complaints = int(len(model.df))
    total_buildings = int(model.df['bbl'].nunique())
    total_zips = int(model.df['incident_zip'].nunique())

    # Best / worst stats from the model
    quick_stats = model.get_quick_overview_stats()

    return render_template(
        "dashboard.html",
        total_complaints=total_complaints,
        total_buildings=total_buildings,
        total_zips=total_zips,
        **quick_stats
    )





@app.route("/map")
def index():
    """Interactive map page"""
    # pass just the list of category names, which the JS expects
    category_names = list(COMPLAINT_CATEGORIES.keys())
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


@app.route("/api/all_buildings", methods=["GET"])
def api_all_buildings():
    """API endpoint to get all building-level scores (for the map view)."""
    categories = request.args.getlist("categories")
    if not categories:
        categories = list(COMPLAINT_CATEGORIES.keys())

    try:
        data = model.get_all_building_scores_by_category(selected_categories=categories)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
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
            recent = building_complaints.tail(10)[
                ['Complaint_Type', 'Category', 'Status', 'Descriptor']
            ].to_dict('records')

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


# We keep this API for now, but the homepage no longer uses it.
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
    app.run(debug=True, port=5000)
