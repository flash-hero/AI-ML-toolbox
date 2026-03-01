from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import os
import shutil
from typing import List, Optional
from toolbox.vis_analyse import DataAnalyzer
from ..schemas import DatasetInfo

router = APIRouter(
    prefix="/visualization",
    tags=["visualization"]
)

# Shared directories (should match main config)
UPLOAD_DIR = "uploaded_data"
PROCESSED_DIR = "processed_data"
STATIC_DIR = "static/plots"

os.makedirs(STATIC_DIR, exist_ok=True)

def get_file_path(filename: str):
    # Check processed first, then uploaded
    path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(path):
        return path
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return path
    raise HTTPException(status_code=404, detail="File not found")

@router.get("/generate/{filename}")
async def generate_visualizations(filename: str):
    """
    Generates standard visualizations (Boxplots, Correlation, etc.) for the file
    and returns the paths to the generated images.
    """
    try:
        file_path = get_file_path(filename)
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        # Create a specific directory for this file's plots to avoid clutter
        file_clean_name = os.path.splitext(filename)[0]
        plot_dir = os.path.join(STATIC_DIR, file_clean_name)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Clean old plots
        # for f in os.listdir(plot_dir):
        #    os.remove(os.path.join(plot_dir, f))

        analyzer = DataAnalyzer(df)
        
        # Generate plots
        created_plots = {}
        
        # 1. Boxplots
        boxplot_path = analyzer.visualize_data(show_boxplots=True, show_pairplot=False, show_pie=False, save_dir=plot_dir)
        if boxplot_path:
             created_plots["boxplots"] = f"/static/plots/{file_clean_name}/boxplots.png"

        # 2. Correlation Heatmap (Continuous)
        analyzer.relation_continuous(target=df.columns[0], save_dir=plot_dir) # Target doesn't matter for valid correlation matrix
        if os.path.exists(os.path.join(plot_dir, "correlation_heatmap.png")):
             created_plots["correlation"] = f"/static/plots/{file_clean_name}/correlation_heatmap.png"

        # 3. Pie Charts (Categorical)
        # We need to know which files were created. The refactored visualize_data returns a list of paths.
        # But for now let's just list the dir content
        
        return {
            "message": "Visualizations generated successfully",
            "plots": created_plots,
            "base_url": f"/static/plots/{file_clean_name}/"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list/{filename}")
async def list_visualizations(filename: str):
    """Returns a list of available plot URLs for a given file."""
    try:
        file_clean_name = os.path.splitext(filename)[0]
        plot_dir = os.path.join(STATIC_DIR, file_clean_name)
        
        if not os.path.exists(plot_dir):
            return {"plots": []}
            
        # Get all png files
        plots = [f"/static/plots/{file_clean_name}/{f}" for f in os.listdir(plot_dir) if f.endswith('.png')]
        return {"plots": plots}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
