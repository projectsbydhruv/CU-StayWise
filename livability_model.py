import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import json
import re


def normalize_text(text: str) -> str:
    """
    Normalize free-text for fuzzy matching:
    - lower case
    - remove punctuation
    - strip ordinal suffixes (104th -> 104)
    - collapse whitespace
    """
    text = str(text).lower()
    text = re.sub(r'[^0-9a-z\s]', ' ', text)
    text = re.sub(r'\b(\d+)(st|nd|rd|th)\b', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_building_number(text: str):
    """
    Extract leading building number from an address string, e.g.:
    '115 west 104 street' -> 115
    Returns int or None.
    """
    m = re.match(r'\s*(\d+)', str(text))
    return int(m.group(1)) if m else None


# Category definitions with weights (used for aggregation, not the final score)
COMPLAINT_CATEGORIES = {
    'Public Safety': {
        'weight': 10,
        'complaints': [
            'Drug Activity', 'Illegal Fireworks', 'Hazardous Materials',
            'Asbestos', 'Lead', 'Construction Lead Dust', 'Scaffold Safety',
            'SAFETY', 'BEST/Site Safety', 'Cranes and Derricks',
            'Non-Emergency Police Matter', 'Disorderly Youth', 'Panhandling',
            'Encampment', 'Homeless Person Assistance', 'Urinating in Public',
            'Drinking', 'Violation of Park Rules', 'Cannabis Retailer', 'Smoking or Vaping', 'Vendor Enforcement',
            'Illegal Posting', 'Posting Advertisement', 'Outdoor Dining',
            'Tobacco or Non-Tobacco Sale', 'Consumer Complaint',
            'Tattooing', 'Cooling Tower', 'Beach/Pool/Sauna Complaint'
        ]
    },
    'Health Hazards': {
        'weight': 7,
        'complaints': [
            'Rodent', 'Food Poisoning', 'Unsanitary Pigeon Condition',
            'UNSANITARY CONDITION', 'Unsanitary Animal Pvt Property',
            'Harboring Bees/Wasps', 'Mosquitoes', 'Indoor Air Quality',
            'Air Quality', 'Mold', 'Water Quality', 'Drinking Water',
            'Building Drinking Water Tank', 'Industrial Waste', 'Dirty Condition', 'Illegal Dumping',
            'Missed Collection',
            'Graffiti', 'Dead Animal', 'Derelict Vehicles', 'Abandoned Vehicle',
            'Abandoned Bike', 'Litter Basket Complaint', 'Litter Basket Request',
            'Recycling Basket Complaint', 'Dumpster Complaint',
            'Residential Disposal Complaint', 'Commercial Disposal Complaint',
            'Institution Disposal Complaint', 'Street Sweeping Complaint',
            'Sanitation Worker or Vehicle Complaint', 'Standing Water', 'Food Establishment', 'Mobile Food Vendor'
        ]
    },
    'Housing Quality': {
        'weight': 10,
        'complaints': [
            'HEAT/HOT WATER', 'PLUMBING', 'WATER LEAK', 'ELECTRIC',
            'APPLIANCE', 'DOOR/WINDOW', 'FLOORING/STAIRS', 'ELEVATOR',
            'PAINT/PLASTER', 'GENERAL', 'Plumbing', 'Electrical',
            'Boilers', 'Water System', 'Indoor Sewage', 'Sewer',
            'OUTSIDE BUILDING', 'Building/Use', 'Non-Residential Heat',
            'Elevator', 'General Construction/Plumbing', 'Street Condition', 'Sidewalk Condition', 'Curb Condition',
            'Street Light Condition', 'Traffic Signal Condition',
            'Root/Sewer/Sidewalk Condition', 'DEP Street Condition',
            'Broken Parking Meter', 'Street Sign - Missing',
            'Street Sign - Damaged', 'Street Sign - Dangling',
            'Bus Stop Shelter Complaint', 'Bus Stop Shelter Placement',
            'Public Payphone Complaint', 'Snow or Ice', 'Traffic', 'Noise', 'Noise - Street/Sidewalk', 'Noise - Park',
            'Noise - Residential', 'Noise - Commercial', 'Noise - Vehicle',
            'Noise - Helicopter', 'Noise - House of Worship'
        ]
    },

    'Parking & Vehicles': {
        'weight': 3,
        'complaints': [
            'Illegal Parking', 'Blocked Driveway', 'Obstruction',
            'For Hire Vehicle Complaint', 'Taxi Complaint', 'Taxi Report',
            'Green Taxi Complaint', 'For Hire Vehicle Report'
        ]
    },
    'Nature & Surroundings': {
        'weight': 2,
        'complaints': [
            'Damaged Tree', 'Dead/Dying Tree', 'Overgrown Tree/Branches',
            'Illegal Tree Damage', 'Uprooted Stump', 'Wood Pile Remaining',
            'New Tree Request', 'Plant', 'Lot Condition', 'Animal in a Park', 'Unleashed Dog', 'Day Care',
            'Illegal Animal Kept as Pet', 'Illegal Animal Sold',
            'Animal-Abuse', 'Pet Shop',
            'Pet Sale', 'Animal Facility - No Permit'
        ]
    },

    'Administrative': {
        'weight': 1,
        'complaints': [
            'Lost Property', 'Found Property', 'Maintenance or Facility',
            'School Maintenance', 'Borough Office', 'Bike Rack', 'Bench',
            'LinkNYC', 'Wayfinding', 'Bike/Roller/Skate Chronic',
            'Special Projects Inspection Team (SPIT)', 'Real Time Enforcement',
            'Emergency Response Team (ERT)', 'Investigations and Discipline (IAD)',
            'Taxi Compliment', 'Incorrect Data', 'Water Conservation'
        ]
    }
}


class LivabilityModel:
    def __init__(self, df, crime_weight=0.0):
        """Livability model based purely on 311 complaints.

        crime_weight is kept for backward compatibility but is ignored – the
        livability score is now driven only by complaint patterns.
        """
        self.df = df.copy()
        # We ignore crime in the score computation
        self.crime_weight = 0.0
        self.complaint_weight = 1.0
        self.current_year = datetime.now().year

        # Prepare complaint categories and mapping
        self._prepare_data()

        # Precompute building-level stats used for livability score
        self._prepare_building_stats()

    def _prepare_data(self):
        if 'bbl' not in self.df.columns:
            print("Warning: BBL column missing, using bin as building identifier")
            self.df['bbl'] = self.df['bin']

        # Create category lookup
        self.complaint_to_category = {}
        for category, data in COMPLAINT_CATEGORIES.items():
            for complaint in data['complaints']:
                self.complaint_to_category[complaint] = category

        # Map complaints to categories
        self.df['Category'] = self.df['Complaint_Type'].map(self.complaint_to_category)

        # Handle unmapped complaints
        unmapped = self.df[self.df['Category'].isna()]
        if len(unmapped) > 0:
            print(f"Warning: {len(unmapped)} complaints not mapped to categories")
            self.df.loc[self.df['Category'].isna(), 'Category'] = 'Administrative'

        # Build a normalized search_text field combining address-related columns
        search_cols = []
        for col in [
            'Incident_Address', 'incident_address',
            'Address', 'address',
            'Street_Name', 'street_name',
            'Landmark', 'landmark',
            'incident_zip'
        ]:
            if col in self.df.columns:
                search_cols.append(col)

        if search_cols:
            combined = self.df[search_cols].astype(str).agg(' '.join, axis=1)
            self.df['search_text'] = combined.apply(normalize_text)
        else:
            # Fallback: at least have something
            if 'Incident_Address' in self.df.columns:
                self.df['search_text'] = self.df['Incident_Address'].astype(str).apply(normalize_text)
            else:
                self.df['search_text'] = ""

    def _prepare_building_stats(self):
        """Precompute building-level statistics for livability score.

        - complaint_count per building
        - max values used for normalization (using 99th percentile)
        The score is now based only on 311 complaints.
        """
        # Use all categories by default
        self.building_scores_all = self.calculate_building_scores_by_category(
            selected_categories=list(COMPLAINT_CATEGORIES.keys())
        )

        if len(self.building_scores_all) == 0:
            # Fallbacks if something goes wrong
            self.max_complaints = 1
            self.max_crime_incidents = 1  # kept for backward compatibility, but unused
            return

        # Use the 99th percentile of complaint_count for normalization
        complaints = self.building_scores_all['complaint_count'].fillna(0)
        pct99 = complaints.quantile(0.99)
        self.max_complaints = max(1, pct99)

        # We no longer use crime in the score, but keep the attribute so
        # existing code that reads it does not break.
        self.max_crime_incidents = 1

    def calculate_zipcode_scores_by_category(self, selected_categories=None):
        """
        Calculate raw complaint counts by zip code for selected categories.
        Also compute the most_common_category for each ZIP for coloring the map.
        """
        if selected_categories is None:
            selected_categories = list(COMPLAINT_CATEGORIES.keys())

        filtered_df = self.df[self.df['Category'].isin(selected_categories)]

        zipcode_data = filtered_df.groupby('incident_zip').agg({
            'Unique_Key': 'count',
            'bbl': 'nunique',
            'latitude': 'mean',
            'longitude': 'mean',
            'Category': lambda x: x.value_counts().index[0] if len(x) > 0 else None
        }).rename(columns={
            'Unique_Key': 'total_complaints',
            'bbl': 'num_buildings',
            'Category': 'most_common_category'
        })

        # per-building complaints
        zipcode_data['complaints_per_building'] = (
                zipcode_data['total_complaints'] / zipcode_data['num_buildings']
        )

        total_complaints_all_zips = zipcode_data['total_complaints'].sum()
        if total_complaints_all_zips > 0:
            zipcode_data['intensity_score'] = (
                    (zipcode_data['total_complaints'] / total_complaints_all_zips) * 100
            ).round(2)
        else:
            zipcode_data['intensity_score'] = 0

        result = zipcode_data.reset_index()

        output_cols = [
            'incident_zip', 'intensity_score', 'num_buildings',
            'total_complaints', 'complaints_per_building',
            'latitude', 'longitude', 'most_common_category'
        ]

        return result[output_cols]

    def calculate_building_scores_by_category(self, selected_categories=None):
        """
        Calculate raw complaint counts by building for selected categories.
        Also compute the most_common_category per building for coloring.
        """
        if selected_categories is None:
            selected_categories = list(COMPLAINT_CATEGORIES.keys())

        filtered_df = self.df[self.df['Category'].isin(selected_categories)]

        building_scores = filtered_df.groupby('bbl').agg({
            'Unique_Key': 'count',
            'incident_zip': 'first',
            'Incident_Address': 'first',
            'latitude': 'first',
            'longitude': 'first',
            'Category': lambda x: x.value_counts().index[0] if len(x) > 0 else None
        }).rename(columns={
            'Unique_Key': 'complaint_count',
            'Category': 'most_common_category'
        })

        building_scores = building_scores.reset_index()

        # intensity within each ZIP
        for zip_code in building_scores['incident_zip'].unique():
            zip_mask = building_scores['incident_zip'] == zip_code
            zip_buildings = building_scores[zip_mask]
            total_complaints_in_zip = zip_buildings['complaint_count'].sum()

            if total_complaints_in_zip > 0:
                building_scores.loc[zip_mask, 'intensity_score'] = (
                        (zip_buildings['complaint_count'] / total_complaints_in_zip) * 100
                ).round(2)
            else:
                building_scores.loc[zip_mask, 'intensity_score'] = 0

        return building_scores

    def get_all_building_scores_by_category(self, selected_categories=None):
        """
        Return all building scores (across all ZIP codes) for the selected categories.
        Used by /api/all_buildings for the city-wide building view.
        """
        all_categories_keys = list(COMPLAINT_CATEGORIES.keys())

        if selected_categories is None or set(selected_categories) == set(all_categories_keys):
            # If all categories are selected, reuse the precomputed building scores
            if hasattr(self, "building_scores_all") and self.building_scores_all is not None:
                return self.building_scores_all.copy().to_dict('records')

        # Otherwise, recompute for the subset
        building_scores = self.calculate_building_scores_by_category(selected_categories)
        return building_scores.to_dict('records')

    def get_category_breakdown(self, zip_code=None, bbl=None):
        """Get complaint category breakdown for a zip code or building"""
        if zip_code:
            subset = self.df[self.df['incident_zip'] == zip_code]
        elif bbl:
            subset = self.df[self.df['bbl'] == bbl]
        else:
            subset = self.df

        breakdown = subset.groupby('Category')['Unique_Key'].count().sort_values(ascending=False)
        return breakdown.to_dict()

    def get_temporal_stats(self, selected_categories=None, zip_code=None):
        """
        Get temporal statistics for selected categories

        Returns yearly trends and basic counts
        """
        if selected_categories:
            filtered_df = self.df[self.df['Category'].isin(selected_categories)]
        else:
            filtered_df = self.df

        if zip_code:
            filtered_df = filtered_df[filtered_df['incident_zip'] == zip_code]

        if 'year_created' in filtered_df.columns:
            yearly = filtered_df.groupby('year_created').size().to_dict()
        else:
            yearly = {}

        return {
            'total_complaints': len(filtered_df),
            'yearly_trend': yearly,
            'categories': filtered_df['Category'].value_counts().to_dict()
        }

    def get_building_livability(self, bbl):
        """Compute a 0–100 livability score for a single building (higher = better).

        The score is now based *only* on 311 complaint patterns:
        - relative complaint volume (within all buildings)

        Higher complaint volume ⇒ higher risk ⇒ lower livability score.
        """
        if self.building_scores_all is None or len(self.building_scores_all) == 0:
            return None

        row = self.building_scores_all[self.building_scores_all['bbl'] == bbl]
        if row.empty:
            return None

        row = row.iloc[0]

        # Complaint risk (0–1). We cap at the 99th percentile value stored
        # in self.max_complaints and apply a square-root transform so that
        # differences at the low end of complaints matter more than extreme outliers.
        raw_complaints = float(row['complaint_count'])
        if self.max_complaints > 0:
            ratio = min(raw_complaints / float(self.max_complaints), 1.0)
        else:
            ratio = 0.0
        complaint_index = ratio ** 0.5  # 0 = best, 1 = worst

        # Combined risk index (0–1), higher = worse (complaints only)
        risk_index = complaint_index

        # Livability score: 0–100, higher = better
        livability_score = max(0.0, min(100.0, (1.0 - risk_index) * 100.0))

        # Category breakdown for this building
        category_breakdown = self.get_category_breakdown(bbl=bbl)

        # Package a clean dictionary for the frontend
        result = {
            'bbl': int(row['bbl']),
            'address': row.get('Incident_Address'),
            'zipcode': int(row['incident_zip']) if not pd.isna(row['incident_zip']) else None,
            'latitude': float(row['latitude']) if not pd.isna(row['latitude']) else None,
            'longitude': float(row['longitude']) if not pd.isna(row['longitude']) else None,
            'complaint_count': int(raw_complaints),
            'total_crime_incidents': None,  # kept for backward compatibility; not used in scoring
            'livability_score': round(livability_score, 1),
            'risk_index': round(risk_index, 3),
            'category_breakdown': category_breakdown
        }

        return result

    def search_address(self, building=None, street=None, zipcode=None, max_results=5):
        """
        Search for buildings matching a combination of:
        - building number
        - street name
        - ZIP code

        Uses fuzzy token matching on a precomputed search_text field.
        If no exact building is found, falls back to the nearest building
        on the same street (by building number), and labels it as such.
        """
        df = self.df

        # Clean inputs
        building = (building or "").strip()
        street = (street or "").strip()
        zipcode = (zipcode or "").strip()

        if not any([building, street, zipcode]):
            return []

        candidates = df

        # 1) Filter by ZIP if provided
        if zipcode and 'incident_zip' in candidates.columns:
            if zipcode.isdigit() and len(zipcode) == 5:
                zip_int = int(zipcode)
                candidates = candidates[candidates['incident_zip'] == zip_int]

        # 2) Filter by street (token-based fuzzy match using search_text)
        street_tokens = []
        if street:
            street_tokens = normalize_text(street).split()

        street_filtered = None
        if street_tokens:
            mask_street = candidates['search_text'].apply(
                lambda txt: all(tok in txt for tok in street_tokens)
            )
            street_filtered = candidates[mask_street]
            if not street_filtered.empty:
                candidates = street_filtered

        # 3) Try to match building number exactly within current candidates
        building_num = extract_building_number(building) if building else None

        exact_building_matches = None
        if building_num is not None and 'Incident_Address' in candidates.columns:
            def same_building(addr):
                return extract_building_number(addr) == building_num

            mask_b = candidates['Incident_Address'].apply(same_building)
            exact_building_matches = candidates[mask_b]
            if not exact_building_matches.empty:
                candidates = exact_building_matches

        results = []

        # 4) If we have any candidates, treat them as exact/close matches
        if not candidates.empty:
            for bbl in candidates['bbl'].dropna().unique()[:max_results]:
                info = self.get_building_livability(bbl)
                if info:
                    info['match_note'] = "Exact or close match for your input."
                    results.append(info)

            results.sort(key=lambda x: x['livability_score'], reverse=True)
            return results

        # 5) Fallback: nearest building on same street by building number
        if building_num is not None and street_tokens:
            base = df

            mask_street_all = base['search_text'].apply(
                lambda txt: all(tok in txt for tok in street_tokens)
            )
            street_only = base[mask_street_all]

            if not street_only.empty and 'Incident_Address' in street_only.columns:
                def distance(row):
                    bn = extract_building_number(row['Incident_Address'])
                    return abs(bn - building_num) if bn is not None else float("inf")

                street_only = street_only.assign(_dist=street_only.apply(distance, axis=1))
                street_only = street_only.sort_values('_dist')
                nearest = street_only.head(max_results)

                for bbl in nearest['bbl'].dropna().unique():
                    info = self.get_building_livability(bbl)
                    if info:
                        info['match_note'] = (
                            "Nearest available building on this street based on our data "
                            "(no exact match found)."
                        )
                        results.append(info)

                results.sort(key=lambda x: x['livability_score'], reverse=True)
                return results

        # If all else fails, return empty list (caller may show generic message)
        return []

    def get_dashboard_summary(self):
        """
        Summary data used on the landing dashboard.

        Returns:
            dict with overall counts, zip-level info and top buildings.
        """
        # Basic overall counts
        total_complaints = int(len(self.df))
        total_buildings = int(self.df['bbl'].nunique())

        # List of zip codes in the data
        zip_list = sorted(self.df['incident_zip'].unique())

        # Most common complaint category in each zip code
        zip_cat = (
            self.df
            .groupby(['incident_zip', 'Category'])['Unique_Key']
            .count()
            .reset_index(name='count')
        )

        # For each zip, keep the category with the highest count
        idx = zip_cat.groupby('incident_zip')['count'].idxmax()
        top_by_zip_df = zip_cat.loc[idx]

        top_complaint_by_zip = (
            top_by_zip_df
            .set_index('incident_zip')[['Category', 'count']]
            .to_dict(orient='index')
        )

        # Use zipcode scores for Public Safety to get a simple "safety" score
        safety_scores = self.calculate_zipcode_scores_by_category(
            selected_categories=['Public Safety']
        )

        if len(safety_scores) > 0:
            max_intensity = safety_scores['intensity_score'].max()
            min_intensity = safety_scores['intensity_score'].min()

            if max_intensity != min_intensity:
                safety_scores['safety_score'] = (
                    (max_intensity - safety_scores['intensity_score']) /
                    (max_intensity - min_intensity)
                )
            else:
                safety_scores['safety_score'] = 1.0

            top_safe_zips = (
                safety_scores
                .sort_values('safety_score', ascending=False)
                .to_dict(orient='records')
            )
        else:
            top_safe_zips = []

        # Buildings with the fewest Housing Quality complaints
        housing_buildings = self.calculate_building_scores_by_category(
            selected_categories=['Housing Quality']
        )

        if len(housing_buildings) > 0:
            # Keep only buildings with at least one housing complaint
            housing_buildings = housing_buildings[
                housing_buildings['complaint_count'] > 0
            ]

            housing_buildings = housing_buildings.sort_values(
                'complaint_count',
                ascending=True
            )

            top_housing_buildings = housing_buildings.head(5)[
                ['bbl', 'Incident_Address', 'incident_zip', 'complaint_count']
            ].to_dict(orient='records')
        else:
            top_housing_buildings = []

        return {
            'total_complaints': total_complaints,
            'total_buildings': total_buildings,
            'zip_list': zip_list,
            'top_complaint_by_zip': top_complaint_by_zip,
            'top_safe_zips': top_safe_zips,
            'top_housing_buildings': top_housing_buildings
        }

    def get_quick_overview_stats(self):
        """
        Compute best / worst ZIPs, streets, and buildings by livability score.

        Livability score here is 0–100 based ONLY on 311 complaint volume:
            score = 100 * (1 - complaint_count / max_complaints)

        Returns a dict with keys:
            best_zip, worst_zip,
            best_street, worst_street,
            best_building, worst_building

        Each key maps to either a small dict or None if there isn't enough data.
        """

        # Make sure we have building-level scores prepared
        if not hasattr(self, "building_scores_all") or self.building_scores_all is None:
            self._prepare_building_stats()

        bs = self.building_scores_all.copy()

        # Basic safety checks – these columns should always exist
        required_cols = ["complaint_count", "incident_zip", "Incident_Address"]
        for col in required_cols:
            if col not in bs.columns:
                raise KeyError(f"Expected column '{col}' in building_scores_all")

        # Fill NA complaints with 0
        bs["complaint_count"] = bs["complaint_count"].fillna(0)

        # Make sure max_complaints is sensible
        max_c = float(bs["complaint_count"].max())
        if max_c <= 0:
            max_c = 1.0

        # Compute livability score: 100 when complaints are 0, decreasing as complaints increase
        bs["livability_score"] = 100.0 * (1.0 - bs["complaint_count"] / max_c)
        bs["livability_score"] = bs["livability_score"].clip(lower=0.0, upper=100.0)

        # ---------------- ZIP-LEVEL ----------------
        zip_stats = (
            bs.groupby("incident_zip")["livability_score"]
              .mean()
              .reset_index()
              .dropna()
        )

        best_zip = worst_zip = None
        if not zip_stats.empty:
            best_zip_row = zip_stats.sort_values("livability_score", ascending=False).iloc[0]
            worst_zip_row = zip_stats.sort_values("livability_score", ascending=True).iloc[0]

            best_zip = {
                "zip": int(best_zip_row["incident_zip"]),
                "avg_score": round(float(best_zip_row["livability_score"]), 1)
            }
            worst_zip = {
                "zip": int(worst_zip_row["incident_zip"]),
                "avg_score": round(float(worst_zip_row["livability_score"]), 1)
            }

        # ---------------- STREET-LEVEL ----------------
        def extract_street(addr):
            if pd.isna(addr):
                return None
            s = str(addr).strip()
            parts = s.split(" ", 1)
            return parts[1].strip() if len(parts) > 1 else None

        bs["street_name"] = bs["Incident_Address"].apply(extract_street)

        street_stats = (
            bs.dropna(subset=["street_name"])
              .groupby("street_name")["livability_score"]
              .mean()
              .reset_index()
        )

        best_street = worst_street = None
        if not street_stats.empty:
            best_street_row = street_stats.sort_values("livability_score", ascending=False).iloc[0]
            worst_street_row = street_stats.sort_values("livability_score", ascending=True).iloc[0]

            best_street = {
                "street": str(best_street_row["street_name"]),
                "avg_score": round(float(best_street_row["livability_score"]), 1)
            }
            worst_street = {
                "street": str(worst_street_row["street_name"]),
                "avg_score": round(float(worst_street_row["livability_score"]), 1)
            }

        # ---------------- BUILDING-LEVEL ----------------
        best_building = worst_building = None
        if not bs.empty:
            # best
            best_row = bs.sort_values("livability_score", ascending=False).iloc[0]
            worst_row = bs.sort_values("livability_score", ascending=True).iloc[0]

            best_building = {
                "address": str(best_row["Incident_Address"]),
                "zip": int(best_row["incident_zip"]),
                "score": round(float(best_row["livability_score"]), 1),
                "complaints": int(best_row["complaint_count"])
            }
            worst_building = {
                "address": str(worst_row["Incident_Address"]),
                "zip": int(worst_row["incident_zip"]),
                "score": round(float(worst_row["livability_score"]), 1),
                "complaints": int(worst_row["complaint_count"])
            }

        return {
            "best_zip": best_zip,
            "worst_zip": worst_zip,
            "best_street": best_street,
            "worst_street": worst_street,
            "best_building": best_building,
            "worst_building": worst_building
        }




def process_data(csv_path, crime_weight=0.0):
    """ Main processing function (not used by Flask app, but kept for testing) """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Initializing model (crime weight: {crime_weight})... (crime ignored in current version)")
    model = LivabilityModel(df, crime_weight=crime_weight)

    print("Calculating scores...")
    zipcode_scores = model.calculate_zipcode_scores_by_category()
    building_scores = model.calculate_building_scores_by_category()

    # Save processed data
    zipcode_scores.to_csv('zipcode_scores.csv', index=False)
    building_scores.to_csv('building_scores.csv', index=False)

    print(f"\nProcessed {len(building_scores)} buildings across {len(zipcode_scores)} zip codes")
    print(f"\nTop 5 Highest Complaint Density Zip Codes:")
    print(zipcode_scores.nlargest(5, 'intensity_score')[
        ['incident_zip', 'intensity_score', 'num_buildings', 'complaints_per_building']])
    print(f"\nLowest 5 Complaint Density Zip Codes:")
    print(zipcode_scores.nsmallest(5, 'intensity_score')[
        ['incident_zip', 'intensity_score', 'num_buildings', 'complaints_per_building']])

    return model, building_scores, zipcode_scores
