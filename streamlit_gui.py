import streamlit as st
import os
import numpy as np
import io
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pprint import pformat

from cobaya.yaml import yaml_dump, yaml_load
from cobaya.bib import pretty_repr_bib, get_bib_info, get_bib_component
from cobaya.tools import get_available_internal_class_names, \
    cov_to_std_and_corr, resolve_packages_path, sort_cosmetic
from cobaya.input import get_default_info
from cobaya.conventions import kinds, packages_path_env, packages_path_input
from cobaya.cosmo_input.input_database import _combo_dict_text
from cobaya.cosmo_input import input_database
from cobaya.cosmo_input.autoselect_covmat import get_best_covmat, covmat_folders
from cobaya.cosmo_input.create_input import create_input


st.set_page_config(
    page_title="Cobaya Input Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

NONE_PRESET_KEY = '(None)'

def text(key, contents):
    """Get description text for a key, or the key itself if no description exists."""
    desc = (contents or {}).get("desc")
    return desc or key

def update_input_and_covmat():
    """Generates Cobaya input and fetches covmat based on current session state selections."""
    try:
        # Use the selections *excluding* the 'preset' key itself for generation
        current_selections = st.session_state.get('selections', {})
        selections_for_create = {k: v for k, v in current_selections.items() if k != 'preset'}

        # Check if we are stuck in an error loop caused by a preset
        if (st.session_state.get('last_error_preset') is not None and
            st.session_state.selections.get('preset') == st.session_state.last_error_preset):
             st.warning(f"Skipping update for preset '{st.session_state.selections.get('preset')}' due to previous error: {st.session_state.last_error}")
             st.session_state.current_info = {"error": f"Failed to generate input for preset '{st.session_state.selections.get('preset')}': {st.session_state.last_error}"}
             st.session_state.current_covmat_data = None
             st.session_state.covmat_message = "Covmat search skipped due to input generation error."
             return

        st.session_state.last_error = None
        st.session_state.last_error_preset = None

        info = create_input(
            get_comments=True,
            **selections_for_create
        )
        st.session_state.current_info = info

      # --- Covmat Logic ---
        packages_path = resolve_packages_path()
        covmat_data = None
        covmat_message = ""

        if not packages_path:
            covmat_message = (
                f"Could not find packages path. Define it via env var '{packages_path_env}' "
                f"or input key '{packages_path_input}' to enable automatic covmat finding."
            )
        elif all(not os.path.isdir(d.format(**{packages_path_input: packages_path}))
                 for d in covmat_folders):
            covmat_message = (
                f"External packages path found ({packages_path}), but expected "
                f"subfolders ({', '.join(covmat_folders)}) not found. "
                "Cannot search for covmats."
            )
        else:
            try:
                info_for_covmat = info.copy() if isinstance(info, dict) else yaml_load(info)
                if isinstance(info_for_covmat, dict):
                     covmat_data = get_best_covmat(info_for_covmat, packages_path=packages_path)
                     covmat_message = f"Best covariance matrix found: '{covmat_data['name']}'"
                else:
                     covmat_message = "Covmat search skipped as input generation did not produce a valid dictionary."

            except Exception as cov_e:
                if isinstance(info, str):
                     covmat_message = "Covmat search skipped due to input generation error."
                else:
                     covmat_message = f"Found packages path, but failed to find/load covmat for current selection: {str(cov_e)}"


        st.session_state.current_covmat_data = covmat_data
        st.session_state.covmat_message = covmat_message
        # --- End Covmat Logic ---

    except Exception as e:
        st.error(f"Error creating input: {str(e)}")
        error_message = f"Failed to generate input: {str(e)}"
        st.session_state.current_info = {"error": error_message}
        st.session_state.current_covmat_data = None
        st.session_state.covmat_message = "Covmat search skipped due to input generation error."
        st.session_state.last_error = str(e)
        st.session_state.last_error_preset = st.session_state.selections.get('preset')


def handle_preset_change():
    """Callback function when the preset selection changes."""
    selected_preset_text = st.session_state.select_preset
    preset_options_dict = getattr(input_database, "preset")
    preset_option_keys = list(preset_options_dict.keys())
    preset_option_texts = [text(k, v) for k, v in preset_options_dict.items()]
    try:
         preset_key_index = preset_option_texts.index(selected_preset_text)
         preset_key = preset_option_keys[preset_key_index]
    except ValueError:
         st.error(f"Internal Error: Could not find the key for selected preset text '{selected_preset_text}'.")
         st.session_state.selections['preset'] = NONE_PRESET_KEY
         st.session_state.trigger_update = True
         return

    st.session_state.selections['preset'] = preset_key

    if preset_key and preset_key != NONE_PRESET_KEY:
        preset_values = input_database.preset.get(preset_key, {})
        for field, value_key in preset_values.items():
            if field in st.session_state.selections and field != 'desc':
                options_dict = getattr(input_database, field, None)
                if options_dict and value_key in options_dict:
                     st.session_state.selections[field] = value_key
                else:
                     st.warning(f"Preset '{preset_key}' requested value '{value_key}' for field '{field}', but this value is not a valid option. Skipping update for this field.")

    st.session_state.trigger_update = True
    st.session_state.last_error = None
    st.session_state.last_error_preset = None


def create_field_callback(field_name):
    """Create a callback function for a specific field."""
    def callback():
        # Get the selected text from session state
        selected_text = st.session_state[f"select_{field_name}"]

        # Get the options for this field
        options_dict = getattr(input_database, field_name)
        option_keys = list(options_dict.keys())
        option_texts = [text(k, v) for k, v in options_dict.items()]

        try:
            # Map the selected text back to the key
            new_selection_key = option_keys[option_texts.index(selected_text)]
        except ValueError:
            st.error(f"Error mapping text '{selected_text}' to key for '{field_name}'.")
            return

        # Update the selection and trigger an update
        if st.session_state.selections[field_name] != new_selection_key:
            st.session_state.selections[field_name] = new_selection_key
            st.session_state.selections['preset'] = NONE_PRESET_KEY
            st.session_state.trigger_update = True
            st.session_state.last_error = None
            st.session_state.last_error_preset = None

    return callback


def save_covmat_txt(params, covmat):
    output = io.StringIO()
    str_params = [str(p) for p in params]
    output.write("# " + " ".join(str_params) + "\n")
    np.savetxt(output, covmat)
    return output.getvalue()

# --- Streamlit GUI ---

def streamlit_gui():
    """Main function for the Streamlit GUI."""
    # Page config is already done!
    # --- Now proceed with the rest of the GUI setup ---
    st.title("Cobaya Input Generator for Cosmology")
    st.markdown("Generate input files (`info` dictionaries) for cosmological analyses with Cobaya.")

    # Initialize session state (safe now)
    if 'selections' not in st.session_state:
        st.session_state.selections = {}
        for _, fields in _combo_dict_text:
            for field, _ in fields:
                 options_dict = getattr(input_database, field)
                 first_key = list(options_dict.keys())[0]
                 st.session_state.selections[field] = first_key
        st.session_state.selections['preset'] = NONE_PRESET_KEY

    # Initialize other state variables
    if 'current_info' not in st.session_state: st.session_state.current_info = None
    if 'current_covmat_data' not in st.session_state: st.session_state.current_covmat_data = None
    if 'covmat_message' not in st.session_state: st.session_state.covmat_message = "Generating initial configuration..."
    if 'last_error' not in st.session_state: st.session_state.last_error = None
    if 'last_error_preset' not in st.session_state: st.session_state.last_error_preset = None
    if 'trigger_update' not in st.session_state: st.session_state.trigger_update = True

    # --- Sidebar for Options ---
    with st.sidebar:
        st.header("Options")
        for group, fields in _combo_dict_text:
            st.subheader(group)
            for field, desc in fields:
                options_dict = getattr(input_database, field)
                option_keys = list(options_dict.keys())
                option_texts = [text(k, v) for k, v in options_dict.items()]

                try:
                    current_selection_key = st.session_state.selections[field]
                    current_index = option_keys.index(current_selection_key)
                except (ValueError, KeyError) as e:
                    st.warning(f"Warning: Could not find key '{st.session_state.selections.get(field)}' in options for '{field}'. Resetting. Error: {e}")
                    current_index = 0
                    if option_keys: st.session_state.selections[field] = option_keys[0]
                    else: st.error(f"Error: No options for field '{field}'!"); st.session_state.selections[field] = None

                if field == "preset":
                    st.selectbox(
                        desc, options=option_texts, index=current_index,
                        key="select_preset", on_change=handle_preset_change
                    )
                else:
                    # Create a callback for this specific field
                    st.selectbox(
                        desc, options=option_texts, index=current_index,
                        key=f"select_{field}", on_change=create_field_callback(field)
                    )

        # The manual_selection_changed flag is no longer needed as we use callbacks

    # --- Main Content Area Logic ---
    if st.session_state.trigger_update:
        update_input_and_covmat()
        st.session_state.trigger_update = False

    # --- Display Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["YAML", "Python", "Bibliography", "Covariance Matrix", "Component Info"])

    # Prepare content (simplified logic)
    current_info = st.session_state.get('current_info', {})
    yaml_content, python_content, bib_content = "Preparing...", "Preparing...", "Preparing..." # Defaults

    if isinstance(current_info, dict) and "error" in current_info:
        error_msg = current_info["error"]
        yaml_content = f"# {error_msg}"
        python_content = f"# {error_msg}"
        bib_content = f"# {error_msg}"
    elif current_info is None:
         yaml_content = "# Generating initial configuration..."
         python_content = "# Generating initial configuration..."
         bib_content = "# Generating initial configuration..."
    elif isinstance(current_info, dict):
        info_copy = current_info.copy()
        comments = info_copy.pop("comment", None)
        comments_text = "\n# " + "\n# ".join(comments) if comments else ""
        try:
            yaml_content = yaml_dump(sort_cosmetic(info_copy)) + comments_text
        except Exception as e: yaml_content = f"# YAML dump error: {e}\n{pformat(info_copy)}"
        python_content = "info = " + pformat(info_copy) + comments_text
        try:
            bib_content = pretty_repr_bib(*get_bib_info(info_copy))
        except Exception as e: bib_content = f"# Bib error: {e}"
    elif isinstance(current_info, str): # Handle direct string error from create_input
         yaml_content = f"# Input generation failed:\n{current_info}"
         python_content = f"# Input generation failed:\n{current_info}"
         bib_content = f"# Input generation failed:\n{current_info}"

    # Display Tab Content (YAML, Python, Bib)
    with tab1:
        st.subheader("Generated YAML Input")
        st.code(yaml_content, language="yaml")
        st.download_button("Download YAML", yaml_content, "cobaya_input.yaml", "text/yaml", key="dl_yaml")
    with tab2:
        st.subheader("Generated Python Input (`info` dict)")
        st.code(python_content, language="python")
        st.download_button("Download Python", python_content, "cobaya_input.py", "text/x-python", key="dl_py")
    with tab3:
        st.subheader("Combined Bibliography")
        st.code(bib_content, language="bibtex")
        st.download_button("Download Bibliography (.bib)", bib_content, "cobaya_bibliography.bib", "text/plain", key="dl_bib")

    # Display Tab Content (Covmat)
    with tab4:
        st.subheader("Best Guessed Covariance Matrix")
        st.info(st.session_state.get('covmat_message', "No covmat info."))
        covmat_data = st.session_state.get('current_covmat_data')
        if covmat_data and isinstance(covmat_data, dict):
            st.markdown(f"**Covariance file:** `{covmat_data.get('name', 'N/A')}`")
            st.markdown("*Note: Cobaya typically finds this automatically.*")
            params = covmat_data.get("params", [])
            covmat = covmat_data.get("covmat", None)
            if covmat is not None and len(params) > 0:
                try:
                    stds, corrmat = cov_to_std_and_corr(covmat)
                    st.markdown("**Numerical Covariance Matrix:**")
                    str_params = [str(p) for p in params]
                    df = pd.DataFrame(covmat, index=str_params, columns=str_params)
                    st.dataframe(df.style.format("{:.4g}"))

                    st.markdown("**Correlation Matrix Heatmap:**")
                    fig, ax = plt.subplots(figsize=(min(10, len(params)*0.8), min(8, len(params)*0.7)))
                    cmap = colormaps.get_cmap("coolwarm_r")
                    im = ax.imshow(corrmat, cmap=cmap, vmin=-1, vmax=1)
                    ax.set_xticks(np.arange(len(params)))
                    ax.set_yticks(np.arange(len(params)))
                    ax.set_xticklabels(str_params, rotation=45, ha="right")
                    ax.set_yticklabels(str_params)
                    plt.colorbar(im, ax=ax, label="Correlation")
                    plt.subplots_adjust(left=0.2, bottom=0.2)
                    st.pyplot(fig)

                    covmat_txt = save_covmat_txt(params, covmat)
                    st.download_button("Download Covmat (.covmat)", covmat_txt,
                                       os.path.basename(covmat_data.get('name', 'cobaya_covmat.covmat')),
                                       "text/plain", key="dl_covmat")
                except Exception as display_e:
                    st.warning(f"Could not display covmat: {display_e}")
                    if covmat is not None and len(params) > 0:
                         try:
                             covmat_txt = save_covmat_txt(params, covmat)
                             st.download_button("Download Raw Covmat", covmat_txt,
                                                os.path.basename(covmat_data.get('name', 'cobaya_covmat.covmat')),
                                                "text/plain", key="dl_covmat_raw_err")
                         except Exception as save_e: st.error(f"Could not format covmat for download: {save_e}")
            else: st.markdown("Covmat data missing or invalid.")
        else: st.markdown("No covmat determined.")

    # --- Component Info Tab ---
    with tab5:
        st.header("Component Defaults and Bibliography")
        col1, col2 = st.columns(2)
        with col1:
            kind_options = list(kinds) if isinstance(kinds, (dict, set)) else kinds
            selected_kind = st.selectbox("Component type", options=kind_options, key="defaults_kind")
        with col2:
            try: components = get_available_internal_class_names(selected_kind) if selected_kind else []
            except Exception as e: st.error(f"Error getting components for '{selected_kind}': {e}"); components = []
            selected_component = st.selectbox("Component", options=components, key="defaults_component", disabled=not components)

        if selected_component:
            st.subheader(f"Information for {selected_kind}: {selected_component}")
            try:
                defaults_yaml = get_default_info(selected_component, selected_kind, return_yaml=True)
                _indent = "  "
                defaults_yaml_str = str(defaults_yaml)
                full_defaults_yaml = (str(selected_kind) + ":\n" + _indent + str(selected_component) + ":\n" +
                                2 * _indent + ("\n" + 2 * _indent).join(defaults_yaml_str.split("\n")))

                defaults_python = "# Could not parse YAML for Python view."
                try:
                    defaults_dict = yaml_load(full_defaults_yaml)
                    defaults_python = pformat(defaults_dict)
                except Exception as parse_e: st.warning(f"YAML parse warning: {parse_e}")

                defaults_bib = get_bib_component(selected_component, selected_kind) or "# No bibliography found."

                d_tab1, d_tab2, d_tab3 = st.tabs(["YAML Defaults", "Python Defaults", "Component Bibliography"])
                with d_tab1: st.code(full_defaults_yaml, language="yaml")
                with d_tab2: st.code(defaults_python, language="python")
                with d_tab3: st.code(defaults_bib, language="bibtex")
            except Exception as e: st.error(f"Error getting defaults for {selected_component} ({selected_kind}): {e}")
        else:
            st.info("Select a component type and component to view its defaults and bibliography.")

def check_and_install_packages():
    """Check if packages directory exists and install required packages if needed."""
    path = resolve_packages_path() or './packages'
    if not os.path.exists(path):
        st.info(f"The packages directory does not exist. Running 'cobaya-install -p {path} planck_2018_lensing.native'...")
        import subprocess
        try:
            result = subprocess.run(
                ["cobaya-install", "-p", path, "planck_2018_lensing.native"],
                capture_output=True,
                text=True,
                check=True
            )
            st.success("Successfully installed required packages.")
            st.info(result.stdout)
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.error(f"Error installing packages: {e}")
            st.code(e.stderr, language="bash")
        except FileNotFoundError:
            st.error("The 'cobaya-install' command was not found. Make sure Cobaya is properly installed.")

def streamlit_gui_script():
    """Entry point for running the Streamlit GUI."""
    check_and_install_packages()
    streamlit_gui()

if __name__ == '__main__':
    streamlit_gui_script() # Run it
