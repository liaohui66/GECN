# data_utils.py
import torch
import json
import ast 
import numpy as np
import pandas as pd 
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import neighbor_list
from ase.atoms import Atom
from ase.data import atomic_numbers
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from jarvis.core.specie import Specie 
from e3nn import o3 
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple


# --- 1. 原子特征编码器 ---
_ATOM_FEATURES_LOOKUP_TABLE: Optional[np.ndarray] = None
_ATOM_FEATURE_DIM: int = -1
_ATOM_ENCODER_INSTANCE: Optional[OneHotEncoder] = None 

def initialize_atom_features(atom_feat_config: Optional[Dict] = None,
                             force_reinit: bool = False
                            ) -> Tuple[np.ndarray, int]:
    global _ATOM_FEATURES_LOOKUP_TABLE, _ATOM_FEATURE_DIM, _ATOM_ENCODER_INSTANCE

    if _ATOM_FEATURES_LOOKUP_TABLE is not None and _ATOM_FEATURE_DIM != -1 and not force_reinit:
        return _ATOM_FEATURES_LOOKUP_TABLE, _ATOM_FEATURE_DIM

    print("Initializing atom features using EATGNN's logic (CRITICAL: NEEDS YOUR EXACT VERIFICATION)...")

    max_elements = 101
    one_hot_max_categories = 6
    magpie_source = 'magpie'

    if atom_feat_config:
        max_elements = atom_feat_config.get("max_elements", max_elements)
        one_hot_max_categories = atom_feat_config.get("one_hot_max_categories", one_hot_max_categories)
        magpie_source = atom_feat_config.get("magpie_source", magpie_source)

    try:
        # 1. Instantiate OneHotEncoder
        try:
            current_encoder = OneHotEncoder(
                max_categories=one_hot_max_categories, 
                sparse_output=False, 
                handle_unknown='error' # Be strict during fitting
            )
        except TypeError:
            current_encoder = OneHotEncoder(
                max_categories=one_hot_max_categories, 
                sparse=False, 
                handle_unknown='error'
            )

        raw_features_list_for_encoder: List[Any] = [] # This will hold data for fit_transform

        # 2. Get raw features using Specie
        print(f"Fetching raw features for elements 1 to {max_elements} using source='{magpie_source}'...")
        for i in range(1, max_elements + 1):
            try:
                # --- CRITICAL POINT: How Specie output is used with OneHotEncoder ---
                # Assuming get_descrp_arr IS A METHOD and needs to be called.
                # And assuming your EATGNN script directly used its output with OHE.
                # This implies either get_descrp_arr returns categorical data, or
                # it was preprocessed before OHE in your EATGNN.
                
                # OPTION A: If get_descrp_arr() returns the features to be OneHotEncoded directly
                # (This is what your EATGNN snippet `fea = [Specie(...).get_descrp_arr ...]` suggests if get_descrp_arr is a property)
                # If get_descrp_arr is a METHOD, you need parentheses:
                atom_descriptor_val = Specie(Atom(i).symbol, source=magpie_source).get_descrp_arr
                                
                if isinstance(atom_descriptor_val, (list, np.ndarray)):
                    raw_features_list_for_encoder.append(list(atom_descriptor_val)) # OHE expects list of samples
                else:
                    print(f"Warning: Descriptor for atom Z={i} (symbol: {Atom(i).symbol}) is type {type(atom_descriptor_val)}. Expected list/array. Using placeholder.")
                    raw_features_list_for_encoder.append(["ERROR_TYPE"]) # Placeholder with list structure
            except Exception as e:
                print(f"Could not get/process raw feature for atom Z={i}: {e}. Adding placeholder.")
                raw_features_list_for_encoder.append([f"ERROR_FETCH_{i}"]) # Placeholder

        if not raw_features_list_for_encoder:
            raise ValueError("No raw features were prepared for the OneHotEncoder. Check Specie/Atom logic.")

        # 3. At this point, `raw_features_list_for_encoder` should be a list of lists/arrays,
        #    where each inner list/array is the set of features for one element
        #    that will be fed into OneHotEncoder.
        #    e.g., [[feat1_H, feat2_H, ...], [feat1_He, feat2_He, ...], ...]
        #    The OneHotEncoder will then be applied to each "column" of these features
        #    if they are deemed categorical.
        #    THIS IS THE MOST UNCERTAIN PART without seeing your exact EATGNN `fea` preprocessing.
        
        fea_for_ohe = raw_features_list_for_encoder # <<< Corrected variable name

        # 4. Fit and transform
        print(f"Fitting OneHotEncoder on {len(fea_for_ohe)} raw feature sets...")
        if not fea_for_ohe: # Should have been caught earlier
            raise ValueError("Cannot fit OneHotEncoder on empty feature list.")

        # --- This is where the core assumption about OHE input matters ---
        # If fea_for_ohe is, e.g., a list of 132-dim Magpie vectors, OHE(max_cat=6)
        # will try to one-hot encode each of the 132 dimensions as if it's a category
        # with at most 6 values. This is likely NOT what you want for continuous Magpie.
        # You would typically select a few specific categorical Magpie features or bin continuous ones.
        print("WARNING: Applying OneHotEncoder directly to `get_descrp_arr` output. "
              "Verify this matches your EATGNN's intended feature engineering for `fea`.")

        transformed_features = current_encoder.fit_transform(fea_for_ohe)
        _ATOM_ENCODER_INSTANCE = current_encoder

        # 5. Assign to global lookup table and get dimension
        _ATOM_FEATURES_LOOKUP_TABLE = transformed_features.astype(np.float32)
        _ATOM_FEATURE_DIM = _ATOM_FEATURES_LOOKUP_TABLE.shape[1]

        # Padding if some elements failed (though handle_unknown='error' might prevent this stage)
        if _ATOM_FEATURES_LOOKUP_TABLE.shape[0] != max_elements:
             print(f"Warning: Lookup table rows ({_ATOM_FEATURES_LOOKUP_TABLE.shape[0]}) "
                   f"mismatch max_elements ({max_elements}). This implies not all elements up to {max_elements} "
                   "were successfully processed and included in `fea_for_ohe` before `fit_transform`.")
             # This padding logic might need adjustment if raw_features_list_for_encoder
             # doesn't correspond 1-to-1 with elements 1 to max_elements.
             # It's better if raw_features_list_for_encoder always has `max_elements` entries,
             # even if some are placeholders.
             # The current loop `for i in range(1, max_elements + 1):` aims for this.
             if _ATOM_FEATURES_LOOKUP_TABLE.shape[0] < max_elements:
                 actual_fit_elements = _ATOM_FEATURES_LOOKUP_TABLE.shape[0]
                 # Create a new full-size table and copy data, assuming OHE was fit on fewer elements
                 new_lookup_table = np.zeros((max_elements, _ATOM_FEATURE_DIM), dtype=np.float32)
                 # This assumes transformed_features corresponds to the first `actual_fit_elements`
                 # if raw_features_list_for_encoder had fewer items than max_elements due to errors.
                 # This part is tricky without knowing exactly how EATGNN handles missing elements for `fea`.
                 # For now, let's assume fit_transform output matches the length of input `fea_for_ohe`.
                 # If `len(fea_for_ohe)` < `max_elements`, then `transformed_features` will also be shorter.

                 # A safer approach is to ensure `fea_for_ohe` always has `max_elements` items,
                 # using placeholders for problematic elements. The current loop does this.
                 # So, `transformed_features` should already have `max_elements` rows.
                 # The warning might trigger if `handle_unknown='ignore'` was used AND new categories
                 # appeared only during `transform` (not `fit_transform`). But we use `error` for fit.

    except Exception as e:
        # ... (fallback logic - this part seems okay) ...
        print(f"ERROR during EATGNN-style atom feature initialization: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to SIMPLIFIED placeholder (OneHot on symbol) due to error.")
        # ... (rest of fallback) ...
        if _ATOM_FEATURES_LOOKUP_TABLE is None or _ATOM_FEATURE_DIM == -1: # Ensure fallback sets these
            try:
                fb_encoder = OneHotEncoder(max_categories=one_hot_max_categories, sparse_output=False, handle_unknown='ignore')
            except TypeError:
                fb_encoder = OneHotEncoder(max_categories=one_hot_max_categories, sparse=False, handle_unknown='ignore')
            fb_symbols = []
            for i in range(1, max_elements + 1): # Ensure list has max_elements items
                try: fb_symbols.append([Atom(i).symbol])
                except: fb_symbols.append([f"UNK{i}"])

            fb_encoder.fit(fb_symbols)
            try:
                _ATOM_FEATURE_DIM = fb_encoder.transform([["H"]]).shape[1]
                if _ATOM_FEATURE_DIM == 0 and hasattr(fb_encoder, 'categories_') and fb_encoder.categories_:
                    _ATOM_FEATURE_DIM = len(fb_encoder.categories_[0])
            except: _ATOM_FEATURE_DIM = 6 # Default from your output
            if _ATOM_FEATURE_DIM <= 0: _ATOM_FEATURE_DIM = 16 # Final fallback
            
            _ATOM_FEATURES_LOOKUP_TABLE = np.zeros((max_elements, _ATOM_FEATURE_DIM), dtype=np.float32)
            for i in range(1, max_elements + 1):
                try:
                    _ATOM_FEATURES_LOOKUP_TABLE[i-1] = fb_encoder.transform([fb_symbols[i-1]])[0] # Use symbol from fb_symbols
                except: pass
            _ATOM_ENCODER_INSTANCE = fb_encoder


    if _ATOM_FEATURES_LOOKUP_TABLE is None or _ATOM_FEATURE_DIM == -1:
        raise RuntimeError("Atom feature initialization failed to produce a lookup table or dimension after all attempts.")

    print(f"Atom features initialized. Lookup table shape: {_ATOM_FEATURES_LOOKUP_TABLE.shape}, Feature dimension: {_ATOM_FEATURE_DIM}")
    return _ATOM_FEATURES_LOOKUP_TABLE, _ATOM_FEATURE_DIM


# --- 2. Helper function for radial cutoff (from EATGNN) ---
def r_cut2D(radial_cutoff_base: float, ase_atoms_obj: Atom) -> float: # Changed 'cell' to 'ase_atoms_obj'
    # This function was already good from your EATGNN script.
    # structure=AseAtomsAdaptor.get_structure(ase_atoms_obj) # Not needed, ase_atoms_obj is ASE Atoms
    cell_matrix = ase_atoms_obj.get_cell(complete=True).array
    if np.all(np.abs(cell_matrix) < 1e-6): # Handle non-periodic systems (molecules)
        return radial_cutoff_base
    
    norms = [np.linalg.norm(cell_matrix[i]) for i in range(3) if np.any(np.abs(cell_matrix[i]) > 1e-6)]
    if not norms: # All cell vectors are zero (e.g. molecule but cell was [0,0,0])
        return radial_cutoff_base
        
    r_cut = max(max(norms), radial_cutoff_base)
    # r_cut=min(r_cut,max_allowable_radius) # Optional cap from EATGNN
    return r_cut

_print_counter = 0
# --- 3. Core function to process one structure and its target ---
def create_pyg_data(
    pymatgen_structure: Structure,
    piezo_tensor_target: Any, # Should be convertible to 3x3x3 tensor
    atom_features_lookup: np.ndarray, # The global `fea` table
    atom_feature_dim: int,            # The global `dim`
    radial_cutoff: float,
    dtype: torch.dtype,
    irreps_edge_attr_str: Optional[str] = "0x0e" # For creating data.edge_attr if needed by model
) -> Data:
    """
    Converts a Pymatgen Structure and its piezo tensor target into a PyG Data object.
    This function incorporates logic from EATGNN's `datatransform`.
    """
    # <<< 引用并修改全局计数器 >>>
    global _print_counter

    ase_atoms = AseAtomsAdaptor.get_atoms(pymatgen_structure)

    # Node features (data.x)
    node_x_list = []
    for atom_symbol_in_structure in ase_atoms.get_chemical_symbols():
        atomic_num = atomic_numbers[atom_symbol_in_structure]
        if 1 <= atomic_num <= atom_features_lookup.shape[0]:
            node_x_list.append(torch.tensor(atom_features_lookup[atomic_num - 1], dtype=dtype))
        else:
            node_x_list.append(torch.zeros(atom_feature_dim, dtype=dtype))
            
    if not node_x_list:
        x = torch.empty((0, atom_feature_dim), dtype=dtype)
    else:
        x = torch.stack(node_x_list)

    # Node positions (data.pos)
    pos = torch.tensor(ase_atoms.get_positions(), dtype=dtype)

    # Lattice (data.lattice)
    lattice = torch.tensor(ase_atoms.get_cell(complete=True).array, dtype=dtype).unsqueeze(0)

    # Edges and shifts (data.edge_index, data.edge_shift)
    effective_cutoff = r_cut2D(radial_cutoff, ase_atoms)
    if len(ase_atoms) > 1:
        ase_atoms_for_nl = ase_atoms.copy()
        ase_atoms_for_nl.set_pbc(True)
        
        edge_src, edge_dst, edge_shift_raw = neighbor_list(
            "ijS", a=ase_atoms_for_nl, cutoff=effective_cutoff, self_interaction=False
        )
        edge_index = torch.stack([
            torch.from_numpy(edge_src), torch.from_numpy(edge_dst)
        ], dim=0).long()
        edge_shift = torch.from_numpy(edge_shift_raw).to(dtype=dtype)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_shift = torch.empty((0, 3), dtype=dtype)
        
    # Target piezoelectric tensor (data.y_piezo)
    y_piezo = torch.tensor(piezo_tensor_target, dtype=dtype)
    if y_piezo.shape != (3, 3, 3):
        if y_piezo.numel() == 27:
            y_piezo = y_piezo.reshape(3, 3, 3)
        else:
            raise ValueError(f"Piezo tensor target from JSON should be 3x3x3 or reshapeable, but got {y_piezo.shape}")

    # <<< 新增：在这里添加诊断打印 >>>
    if _print_counter < 5: # 只打印前5个样本的 y_piezo 信息
        print(f"\n--- [DEBUG] Sample {_print_counter + 1} Target (y_piezo) Stats ---")
        print(f"Shape: {y_piezo.shape}")
        # 使用 .item() 将tensor标量转换为python数字，避免打印设备信息
        print(f"Mean: {y_piezo.mean().item():.4e}")
        print(f"Std Dev: {y_piezo.std().item():.4e}")
        print(f"Max Value: {y_piezo.max().item():.4e}")
        print(f"Min Value: {y_piezo.min().item():.4e}")
        print("--------------------------------------------------")
        _print_counter += 1

    # Optional: edge_attr
    data_edge_attr = None
    if irreps_edge_attr_str and irreps_edge_attr_str != "0x0e":
        edge_attr_dim = o3.Irreps(irreps_edge_attr_str).dim
        if edge_attr_dim > 0:
            if edge_index.shape[1] > 0:
                data_edge_attr = torch.zeros(edge_index.shape[1], edge_attr_dim, dtype=dtype)
            else:
                data_edge_attr = torch.empty((0, edge_attr_dim), dtype=dtype)

    # Assemble Data object
    data_params = {
        'x': x, 'pos': pos, 'edge_index': edge_index,
        'lattice': lattice, 'edge_shift': edge_shift,
        'y_piezo': y_piezo
    }
    if data_edge_attr is not None:
        data_params['edge_attr'] = data_edge_attr
        
    pyg_data = Data(**data_params)
    return pyg_data

# --- 4. Functions to parse your special concatenated file ---
def parse_concatenated_file(
    file_path: str,
    structure_end_line: int,
    piezo_start_line: int,
    piezo_end_line: int
) -> Tuple[List[Dict[str, Any]], List[Any]]: # 更精确的类型提示
    """
    Parses the special concatenated file format for structures and piezo tensors.
    """
    structures_as_dicts: List[Dict[str, Any]] = []
    piezo_tensors_as_lists: List[Any] = [] # 通常是 List[List[List[float]]]

    print(f"Reading concatenated file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return [], [] # Return empty lists if file not found
    except Exception as e:
        print(f"ERROR: Could not read file {file_path}. Error: {e}")
        return [], []


    # Parse Pymatgen Structure JSON objects
    print(f"Parsing structures (lines 1 to {structure_end_line})...")
    current_object_str_buffer = ""
    brace_counter = 0
    in_object_currently = False # Track if we are inside a { } block

    for i in range(min(len(lines), structure_end_line)): # Ensure we don't go out of bounds
        line = lines[i]
        
        # Heuristic to find start of a new JSON object if not already in one
        if not in_object_currently and line.strip().startswith("{"):
            in_object_currently = True
            current_object_str_buffer = line # Start new buffer with this line
            brace_counter = line.count("{") - line.count("}")
            if brace_counter == 0 and current_object_str_buffer.strip().endswith("}"): # Single line object
                 # Process immediately
                 try:
                    obj_to_parse = current_object_str_buffer.strip()
                    if obj_to_parse.endswith(','): obj_to_parse = obj_to_parse[:-1]
                    structures_as_dicts.append(json.loads(obj_to_parse))
                 except json.JSONDecodeError as e:
                    print(f"Error decoding single-line structure JSON near line {i+1}: {e}")
                 current_object_str_buffer = ""
                 in_object_currently = False
            continue # Move to next line

        if in_object_currently:
            current_object_str_buffer += line
            brace_counter += line.count("{")
            brace_counter -= line.count("}")
            
            if brace_counter == 0: # End of a multi-line object
                try:
                    obj_to_parse = current_object_str_buffer.strip()
                    if obj_to_parse.endswith(','): obj_to_parse = obj_to_parse[:-1]
                    structures_as_dicts.append(json.loads(obj_to_parse))
                except json.JSONDecodeError as e:
                    print(f"Error decoding multi-line structure JSON object ending near line {i+1}: {e}")
                    # print(f"Problematic string part for structure: {current_object_str_buffer[:500]}...")
                current_object_str_buffer = ""
                in_object_currently = False # Reset for next potential object

    # Parse Piezoelectric Tensor list-like strings
    print(f"Parsing piezo tensors (lines {piezo_start_line} to {piezo_end_line})...")
    current_tensor_str_buffer = ""
    bracket_counter = 0
    in_tensor_currently = False # Track if we are inside a [ ] block

    # piezo_start_line is 1-based, lines is 0-based
    for i in range(min(len(lines), piezo_start_line - 1), min(len(lines), piezo_end_line)):
        line = lines[i]

        if not in_tensor_currently and line.strip().startswith("["):
            in_tensor_currently = True
            current_tensor_str_buffer = line
            bracket_counter = line.count("[") - line.count("]")
            if bracket_counter == 0 and current_tensor_str_buffer.strip().endswith("]"): # Single line tensor
                try:
                    tensor_to_eval = current_tensor_str_buffer.strip()
                    if tensor_to_eval.endswith(','): tensor_to_eval = tensor_to_eval[:-1]
                    piezo_tensors_as_lists.append(ast.literal_eval(tensor_to_eval))
                except Exception as e:
                    print(f"Error evaluating single-line piezo tensor near line {i+1}: {e}")
                current_tensor_str_buffer = ""
                in_tensor_currently = False
            continue

        if in_tensor_currently:
            current_tensor_str_buffer += line
            bracket_counter += line.count("[")
            bracket_counter -= line.count("]")

            if bracket_counter == 0: # End of a multi-line tensor
                try:
                    tensor_to_eval = current_tensor_str_buffer.strip()
                    if tensor_to_eval.endswith(','): tensor_to_eval = tensor_to_eval[:-1]
                    piezo_tensors_as_lists.append(ast.literal_eval(tensor_to_eval))
                except Exception as e:
                    print(f"Error evaluating multi-line piezo tensor string ending near line {i+1}: {e}")
                    # print(f"Problematic string part for piezo: {current_tensor_str_buffer[:500]}...")
                current_tensor_str_buffer = ""
                in_tensor_currently = False

    if len(structures_as_dicts) != len(piezo_tensors_as_lists):
        print(f"Warning: Mismatch in parsed structures ({len(structures_as_dicts)}) and piezo tensors ({len(piezo_tensors_as_lists)}).")
    
    return structures_as_dicts, piezo_tensors_as_lists


# --- 5. Main function to load the dataset (ENTRY POINT for other scripts) ---
def load_piezo_dataset( # 这个函数名是 train.py 正在调用的
    large_json_file_path: str,         # train.py 会传递 CONCATENATED_JSON_FILE
    structure_key_in_json: str,    # train.py 会传递 "structure" (或你确认的键名)
    piezo_tensor_key_in_json: str, # train.py 会传递 "total" (或你确认的键名)
    radial_cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
    atom_feat_config: Optional[Dict] = None,
    irreps_edge_attr_for_data: Optional[str] = "0x0e",
    limit_n: Optional[int] = None
) -> List[Data]:
    # 原子特征初始化 (确保它被正确调用并使用 atom_feat_config)
    afe_params = atom_feat_config if atom_feat_config else {}
    atom_features_table, atom_feat_dim = initialize_atom_features(atom_feat_config=afe_params) # 确保这里传递的是字典
    if atom_features_table is None or atom_feat_dim == -1:
        raise RuntimeError("Atom features could not be initialized properly.")

    print(f"Loading large JSON file: {large_json_file_path}")
    data_dict = None
    try:
        with open(large_json_file_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f) # 加载整个文件为一个大字典
    except json.JSONDecodeError as e:
        print(f"FATAL: Could not decode the main JSON file '{large_json_file_path}'. Error: {e}")
        try: # 尝试打印文件开头帮助调试
            with open(large_json_file_path, 'r', encoding='utf-8') as f_err:
                print("\n--- Start of file (first ~500 chars) that caused error ---")
                print(f_err.read(500)); print("--- End of sample content ---")
        except: pass
        return []
    except Exception as e_open:
        print(f"FATAL: Could not open or read main JSON file '{large_json_file_path}'. Error: {e_open}")
        return []

    if not isinstance(data_dict, dict):
        print(f"ERROR: Loaded data from '{large_json_file_path}' is not a dictionary (type: {type(data_dict)}).")
        return []

    # 检查键是否存在
    if structure_key_in_json not in data_dict:
        raise KeyError(f"Structure key '{structure_key_in_json}' not found in JSON. Available keys: {list(data_dict.keys())}")
    if piezo_tensor_key_in_json not in data_dict:
        raise KeyError(f"Piezo tensor key '{piezo_tensor_key_in_json}' not found in JSON. Available keys: {list(data_dict.keys())}")

    structure_dicts_list = data_dict[structure_key_in_json]
    piezo_tensors_list = data_dict[piezo_tensor_key_in_json]

    if not isinstance(structure_dicts_list, list):
        raise TypeError(f"Data under key '{structure_key_in_json}' is not a list (is {type(structure_dicts_list)}).")
    if not isinstance(piezo_tensors_list, list):
        raise TypeError(f"Data under key '{piezo_tensor_key_in_json}' is not a list (is {type(piezo_tensors_list)}).")

    # 打印加载到的数量，用于调试
    print(f"Found {len(structure_dicts_list)} structures under key '{structure_key_in_json}'.")
    print(f"Found {len(piezo_tensors_list)} piezo tensors under key '{piezo_tensor_key_in_json}'.")

    if len(structure_dicts_list) != len(piezo_tensors_list):
        min_len = min(len(structure_dicts_list), len(piezo_tensors_list))
        print(f"Warning: Mismatch in number of structures ({len(structure_dicts_list)}) "
              f"and piezo tensors ({len(piezo_tensors_list)}).")
        if limit_n is None and min_len == 0 : # 如果没有限制且最短为0，则是一个严重问题
             raise ValueError("Data mismatch and no limit_n specified, and one list is empty. Please check the JSON keys or file content.")
        print(f"Will process up to the minimum available {min_len} pairs or limit_n.")
        num_entries_to_process = min_len
    else:
        num_entries_to_process = len(structure_dicts_list)

    if limit_n is not None and limit_n >= 0:
        num_entries_to_process = min(num_entries_to_process, limit_n)
    
    if num_entries_to_process == 0:
        print("No entries to process from the loaded JSON data.")
        return []

    dataset = []
    print(f"Converting {num_entries_to_process} entries to PyG Data objects...")
    for i in tqdm(range(num_entries_to_process)):
        struct_dict = structure_dicts_list[i]
        piezo_data = piezo_tensors_list[i]
        try:
            if not isinstance(struct_dict, dict):
                raise TypeError(f"Structure entry {i} is not a dict, but {type(struct_dict)}")
            pymatgen_structure = Structure.from_dict(struct_dict)
            pyg_data_obj = create_pyg_data(
                pymatgen_structure, piezo_data,
                atom_features_table, atom_feat_dim,
                radial_cutoff, dtype,
                irreps_edge_attr_for_data
            )
            dataset.append(pyg_data_obj)
        except Exception as e:
            mat_id = "Unknown"
            if isinstance(struct_dict, dict): mat_id = struct_dict.get('material_id', f'entry_index_{i}')
            print(f"Error creating Data object for entry {i} (ID: {mat_id}): {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed error
    return dataset
