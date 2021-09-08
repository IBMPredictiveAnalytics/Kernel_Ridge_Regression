# ***********************************************************************
# * Licensed Materials - Property of IBM
# *
# * IBM SPSS Products: Statistics Common
# *
# * (C) Copyright IBM Corp. 1989, 2021
# *
# * US Government Users Restricted Rights - Use, duplication or disclosure
# * restricted by GSA ADP Schedule Contract with IBM Corp.
# ************************************************************************

from wrapper.basewrapper import *
from wrapper import wraputil
from util.statjson import *

from sklearn.kernel_ridge import KernelRidge

"""Initialize the krr wrapper"""
init_wrapper("krr", os.path.join(os.path.dirname(__file__), "KRR-properties.json"))

tableBaseDimensions = ["alpha", "gamma", "coef0", "degree"]


def execute(iterator_id, data_model, settings, lang="en"):
    fields = data_model["fields"]
    check_settings(settings, fields)

    output_json = StatJSON(get_name())
    intl = get_lang_resource(lang)

    cov_size = len(get_value("covariates"))
    kernel_default = {"additive_chi2": {"alpha": 1},
                      "chi2": {"alpha": 1, "gamma": 1},
                      "cosine": {"alpha": 1},
                      "laplacian": {"alpha": 1, "gamma": 1 / cov_size},
                      "linear": {"alpha": 1},
                      "polynomial": {"alpha": 1, "gamma": 1 / cov_size, "coef0": 1, "degree": 3},
                      "rbf": {"alpha": 1, "gamma": 1 / cov_size},
                      "sigmoid": {"alpha": 1, "gamma": 1 / cov_size, "coef0": 1}}

    def has_degree(values):
        return "degree" in values

    def get_kernel_default(kernel_name, kernel_param):
        val = kernel_default[kernel_name].get(kernel_param)
        if val is not None:
            return [val]
        else:
            return None

    def _execute_krr(data):
        kernels = get_value("kernels")
        if data is not None:
            case_count = len(data)
        else:
            return

        records = RecordData(data)

        for field in fields:
            records.add_type(field["type"])

        y_name = get_value("dependent")
        w_name = get_value("weight_var")
        x_names = get_value("covariates")

        columns_data = records.get_columns()

        y_index = wraputil.get_index(fields, y_name)
        w_index = wraputil.get_index(fields, w_name)
        if w_index >= 0:
            if y_index < w_index:
                w, y = columns_data.pop(w_index), columns_data.pop(y_index)
            else:
                y, w = columns_data.pop(y_index), columns_data.pop(w_index)
        else:
            y = columns_data.pop(y_index)
            w = None

        X = list(zip(*columns_data))

        plot_observed = get_value("plot_observed")
        plot_residual = get_value("plot_residual")
        save_pred = is_set("save_pred_newvar")
        save_resid = is_set("save_resid_newvar")
        save_dual = is_set("save_dual_newvar")

        is_use_resid = plot_residual or save_resid
        is_use_predict = plot_observed or save_pred or is_use_resid

        is_saved = save_pred or save_resid or save_dual

        if get_value("crossvalid"):
            from sklearn.model_selection import GridSearchCV
            krrd = KernelRidge()
            param_grid = []

            kernel_functions = {"kernel": process_kernel_name}
            for kernel in kernels:
                trans = {}
                for key in kernel:
                    fun = kernel_functions.get(key, process_vlth_struc)
                    val = fun(kernel[key])
                    if val is None:
                        val = get_kernel_default(kernel["kernel"], key)
                    if val is not None:
                        trans[key] = val

                param_grid.append(trans)

            krr = GridSearchCV(krrd, param_grid, cv=get_value("crossvalid_nfolds"))
            if w is not None:
                krr.fit(X, y, sample_weight=w)
            else:
                krr.fit(X, y)

            nfolds = krr.n_splits_

            pred = None
            resid = None

            if w is not None or is_use_predict:
                pred = krr.predict(X).tolist()
                if is_use_resid:
                    resid = list(map(lambda r: r[0]-r[1], zip(y, pred)))

            if is_saved:
                create_new_variables(case_count, pred, resid)

            if w is not None:
                from sklearn.metrics import r2_score
                r2 = r2_score(y, pred, sample_weight=w)
            else:
                r2 = krr.score(X, y)

            bast_pars = krr.best_params_
            r2_best = krr.best_score_

            best_model_summary_table = Table(intl.loadstring("best_model_summary"), intl.loadstring("mode_summary"))
            best_model_summary_table.update_title(footnote_refs=[0, 1])
            best_model_summary_table.set_default_cell_format(decimals=3)

            best_model_summary_table.add_dimension(Table.DimensionType.ROWS, intl.loadstring("kernel"), True,
                                                   [convert_str(bast_pars["kernel"])])

            best_col_dim_keys = [key for key in tableBaseDimensions if bast_pars.get(key) is not None]
            best_cells = [bast_pars[key] for key in best_col_dim_keys]

            best_cells.extend([nfolds, r2_best, r2])
            best_col_dims = [convert_str(v) for v in best_col_dim_keys]
            
            best_col_dims.extend([intl.loadstring("number_of_crossvalidation_folds"),
                                      intl.loadstring("mean_test_subset_r_square"),
                                      intl.loadstring("full_data_r_square")])

            if has_degree(best_col_dim_keys):
                degree_index = best_col_dims.index(convert_str("degree"))
                if isinstance(best_cells[degree_index], int):
                    degree_dim = Table.Cell(convert_str("degree"))
                    degree_dim.set_default_cell_format(decimals=0)
                    best_col_dims[degree_index] = degree_dim.get_value()

            best_model_summary_table.add_dimension(Table.DimensionType.COLUMNS,
                                                   intl.loadstring("statistics"),
                                                   False,
                                                   best_col_dims)

            best_model_summary_table.add_cells(best_cells)
            best_model_summary_table.add_footnotes(intl.loadstring("dependent_variable").format(y_name))
            best_model_summary_table.add_footnotes(intl.loadstring("model").format(", ".join(x_names)))

            output_json.add_table(best_model_summary_table)

            print_value = get_value("print")
            if print_value != "best":
                results = krr.cv_results_
                params = results['params']
                mean_r2s = results['mean_test_score']
                ranks = results['rank_test_score']

                is_compare = print_value == "compare"
                split_score = {}

                if not is_compare:
                    for fold in range(0, nfolds):
                        name = 'split' + str(fold) + '_test_score'
                        split_score["split" + str(fold + 1)] = results[name]

                model_comparisons_table = Table(intl.loadstring("model_comparisons"),
                                                intl.loadstring("model_comparisons"))

                model_comparisons_table.set_default_cell_format(decimals=3)
                if is_compare:
                    model_comparisons_table.update_title(footnote_refs=[0, 1, 2])
                else:
                    model_comparisons_table.update_title(footnote_refs=[0, 1])

                row_dimensions = []
                comparisons_col_dim_keys = [key for key in tableBaseDimensions
                                            if "_".join(["param", key]) in results.keys()]
                is_integer = True
                for rank in ranks:
                    par = params[rank - 1]
                    row_dimensions.append(convert_str(par["kernel"]))
                    row_cells = [par.get(key, None) for key in comparisons_col_dim_keys]
                    row_cells.append(mean_r2s[rank - 1])
                    if not is_compare:
                        for split in split_score:
                            row_cells.append(split_score[split][rank - 1])
                    model_comparisons_table.add_cells(row_cells)

                if has_degree(comparisons_col_dim_keys):
                    for v in results["param_degree"]:
                        if isinstance(v, float):
                            is_integer = False
                            break

                model_comparisons_table.add_dimension(Table.DimensionType.ROWS, intl.loadstring("kernel"), True,
                                                      row_dimensions)

                comparisons_col_dim_keys.append(intl.loadstring("mean_test_subset_r_square"))
                comparisons_col_dims = [convert_str(k) for k in comparisons_col_dim_keys]
                if not is_compare:
                    for fold in range(len(split_score)):
                        comparisons_col_dims.append(intl.loadstring("fold_r_square").format(fold + 1))

                if has_degree(comparisons_col_dim_keys):
                    degree_index = comparisons_col_dims.index(convert_str("degree"))
                    if is_integer:
                        degree_dim = Table.Cell(convert_str("degree"))
                        degree_dim.set_default_cell_format(decimals=0)
                        comparisons_col_dims[degree_index] = degree_dim.get_value()

                model_comparisons_table.add_dimension(Table.DimensionType.COLUMNS,
                                                      intl.loadstring("statistics"),
                                                      False,
                                                      comparisons_col_dims)

                model_comparisons_table.add_footnotes(intl.loadstring("dependent_variable").format(y_name))
                model_comparisons_table.add_footnotes(intl.loadstring("model").format(", ".join(x_names)))
                if is_compare:
                    model_comparisons_table.add_footnotes(
                        intl.loadstring("number_of_crossvalidation_folds_with_number").format(nfolds))

                output_json.add_table(model_comparisons_table)
        else:
            kernels = kernels[0]

            args = kernel_default[kernels["kernel"]]
            for arg in kernels:
                args[arg] = kernels[arg]

            for key in args:
                if isinstance(args[key], (list, tuple)):
                    args[key] = args[key][0]
            krr = KernelRidge(**args)

            krr.fit(X, y, w)
            r2 = krr.score(X, y, w)

            pred = None
            resid = None

            if is_use_predict:
                pred = krr.predict(X).tolist()
                if is_use_resid:
                    resid = list(map(lambda r: r[0]-r[1], zip(y, pred)))

            if is_saved:
                dual = None
                if is_set("save_dual_newvar"):
                    dual = krr.dual_coef_.tolist()
                create_new_variables(case_count, pred, resid, dual)

            pars = krr.get_params()

            model_summary_table = Table(intl.loadstring("mode_summary"), intl.loadstring("mode_summary"))
            model_summary_table.update_title(footnote_refs=[0, 1])
            model_summary_table.set_default_cell_format(decimals=3)
            model_summary_table.add_dimension(Table.DimensionType.ROWS, intl.loadstring("kernel"), True,
                                              [convert_str(pars["kernel"])])

            col_dim_keys = [key for key in tableBaseDimensions if pars.get(key) is not None]
            cells = [pars[key] for key in col_dim_keys]
            col_dims = [convert_str(v) for v in col_dim_keys]
            col_dims.append(intl.loadstring("r_square"))
            cells.append(r2)

            if has_degree(col_dim_keys):
                degree_index = col_dims.index(convert_str("degree"))
                if isinstance(cells[degree_index], int):
                    degree_dim = Table.Cell(convert_str("degree"))
                    degree_dim.set_default_cell_format(decimals=0)
                    col_dims[degree_index] = degree_dim.get_value()

            model_summary_table.add_dimension(Table.DimensionType.COLUMNS, intl.loadstring("statistics"),
                                              False, col_dims)
            model_summary_table.add_cells(cells)

            model_summary_table.add_footnotes([intl.loadstring("dependent_variable").format(y_name),
                                               intl.loadstring("model").format(", ".join(x_names))])

            output_json.add_table(model_summary_table)

        if get_value("plot_observed"):
            observed_chart = Chart(intl.loadstring("dependent_by_predicted_value").format(y_name),
                                   intl.loadstring("dependent_by_predicted_value").format(y_name))
            observed_chart.set_type(Chart.Type.Scatterplot)
            observed_chart.set_axis_data(Chart.Axis.X, pred)
            observed_chart.set_axis_label(Chart.Axis.X, intl.loadstring("predicted_value"))
            observed_chart.set_axis_data(Chart.Axis.Y, y)
            observed_chart.set_axis_label(Chart.Axis.Y, y_name)

            output_json.add_chart(observed_chart)

        if get_value("plot_residual"):
            residual_chart = Chart(intl.loadstring("residual_by_predicted_value"),
                                   intl.loadstring("residual_by_predicted_value"))
            residual_chart.set_type(Chart.Type.Scatterplot)
            residual_chart.set_axis_data(Chart.Axis.X, pred)
            residual_chart.set_axis_label(Chart.Axis.X, intl.loadstring("predicted_value"))
            residual_chart.set_axis_data(Chart.Axis.Y, resid)
            residual_chart.set_axis_label(Chart.Axis.Y, intl.loadstring("residual"))

            output_json.add_chart(residual_chart)

        generate_output(output_json.get_json(), None)

        finish()

    get_records(iterator_id, data_model, _execute_krr)
    return 0


def process_kernel_name(value):
    return [value]


def process_vlth_struc(value):
    if isinstance(value, dict):
        result = value.get("value_list", [])
        if "low" in value:
            low = value["low"]
            high = value["high"]
            increment = value["increment"]

            def get_float_list(start, stop, step):
                float_list = []
                count = 0
                while True:
                    temp = start + count * step
                    temp = round(temp, 2)
                    if step > 0 and temp > stop:
                        break
                    if step < 0 and temp < stop:
                        break
                    count += 1
                    float_list.append(temp)

                return float_list

            result.extend(get_float_list(low, high, increment))
            result = list(set(result))
    elif isinstance(value, (int, float)):
        result = [value]
    else:
        result = None

    return result


def convert_str(value):
    upper_list = ["rbf"]
    if value in upper_list:
        return value.upper()
    else:
        return value.capitalize()


def create_new_variables(case_count, pred, resid, dual=None):
    fields = []
    new_records = RecordData()

    if is_set("save_pred_newvar"):
        fields.append(create_new_field(get_value("save_pred_newvar"), "double", Measure.SCALE, Role.INPUT))
        new_records.add_columns(pred)
        new_records.add_type(RecordData.CellType.DOUBLE)

    if is_set("save_resid_newvar"):
        fields.append(create_new_field(get_value("save_resid_newvar"), "double", Measure.SCALE, Role.INPUT))
        new_records.add_columns(resid)
        new_records.add_type(RecordData.CellType.DOUBLE)

    if is_set("save_dual_newvar"):
        fields = [create_new_field(get_value("save_dual_newvar"), "double", Measure.SCALE, Role.INPUT)]
        new_records.add_columns(dual)
        new_records.add_type(RecordData.CellType.DOUBLE)

    put_variables(fields, case_count, 0, new_records.get_binaries(), None)
