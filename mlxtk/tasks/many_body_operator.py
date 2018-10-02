import gzip
import io
import pickle

from QDTK.Operatorb import OCoef as Coeff
from QDTK.Operatorb import Operatorb as Operator
from QDTK.Operatorb import OTerm as Term


def create_many_body_operator(name, dofs, grids, coefficients, terms, table):
    if not isinstance(table, str):
        table = "\n".join(table)

    path_pickle = name + ".mb_opr_pickle"

    def task_write_parameters():
        def action_write_parameters(targets):
            for term in terms:
                if isinstance(terms[term], dict):
                    if terms[term]["td"]:
                        terms[term]["td_switch"] = terms[term].get("td_switch", [0])
            obj = [name, dofs, grids, coefficients, terms, table]

            with open(targets[0], "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name": "create_many_body_operator:{}:write_parameters".format(name),
            "actions": [action_write_parameters],
            "targets": [path_pickle],
        }

    def task_write_operator():
        path = name + ".mb_opr.gz"

        def action_write_operator(targets):
            op = Operator()
            op.define_dofs_and_grids(dofs, [grid.get() for grid in grids])

            for coeff in coefficients:
                op.addLabel(coeff, Coeff(coefficients[coeff]))

            for term in terms:
                if isinstance(terms[term], dict):
                    term_dict = terms[term]
                    term_kwargs = {}
                    if "td" in term_dict:
                        terms[term]["td_switch"] = terms[term].get("td_switch", [0])
                        if term_dict["type"] != "diag":
                            raise ValueError(
                                'Only time-depdent terms of type "diag" are supported (not "{}")'.format(
                                    term_dict["type"]
                                )
                            )
                        term_kwargs["tf_label"] = term_dict["td_name"]
                        term_kwargs["tf_args"] = term_dict["td_args"]
                        term_kwargs["tf_switch"] = term_dict["td_switch"]
                    term_kwargs["is_fft"] = term.get("is_fft", False)
                    op.addLabel(term, Term(**term_kwargs))
                else:
                    op.addLabel(term, Term(terms[term]))

            op.readTableb(table)

            with gzip.open(targets[0], "wb") as fp:
                with io.StringIO() as sio:
                    op.createOperatorFileb(sio)
                    fp.write(sio.getvalue().encode())

        return {
            "name": "create_many_body_operator:{}:write_operator".format(name),
            "actions": [action_write_operator],
            "targets": [path],
            "file_dep": [path_pickle],
        }

    return [task_write_parameters, task_write_operator]
