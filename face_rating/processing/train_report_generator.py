# Created by: rschucker
# Date: 2/6/18
# ------------------------------

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import base64
from io import BytesIO


def generate_train_report(train_errors, val_errors, title, training_parameters, moving_average_train_error=3):
    html_strs = []
    html_strs.append('''
        <!DOCTYPE html>
        <html><body>
            <title>{}</title>
            <h2>{}</h2>
            <hr>
        '''.format(title, title)
    )
    html_strs.append('''
            <h3>Model training parameters</h3>
            <table>''')
    print(training_parameters)
    for k, v in sorted(training_parameters.items()):
        html_strs.append('<tr><td>{} = </td><td>{}</td></tr>\n'.format(k, v))

    html_strs.append('''
            </table>
            <hr>
    ''')

    html_strs.append('''
            <h3>Training results</h3>
            <table>
               <tr><td>Final train err = </td><td>{:3.3f}</td></tr>
               <tr><td>Final val err = </td><td>{:3.3f}</td></tr>
            </table>
            <hr>
            <h3>Learning Curve</h3>
        '''.format(np.array(train_errors[-moving_average_train_error:]).mean(),
                   np.array(val_errors[-moving_average_train_error:]).mean())
    )

    html_strs.append(create_html_plot(train_errors, val_errors, title))

    html_strs.append('''
        </body></html>
        ''')

    return ''.join(html_strs)


def create_html_plot(train_errors, val_errors, title):
    plt.figure(figsize=(12, 7))

    plt.plot(np.arange(len(train_errors)), np.array(train_errors), c='g')
    plt.plot(np.arange(len(val_errors)), np.array(val_errors), c='b')

    plt.title('train loss (green) and val loss (blue) for run: {}'.format(title))
    plt.ylim([0.0, max(train_errors + val_errors)])
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    return '<img src="data:image/png;base64,{}"\>\n'.format(
        base64.b64encode(figfile.getvalue()).decode()
    )


def main():
    train_errors = [0.5501488194763295, 0.4142961755960341, 0.36482191596503943, 0.34708897267882066, 0.3376644546282335, 0.3358397224833653, 0.3362512562236996, 0.34269425218224964, 0.3301118746892329, 0.3275222280034237, 0.32452238554470103, 0.3273621913560893, 0.3261449771455627, 0.32153700870796315, 0.3307050257553289, 0.32189768374039174, 0.32645676320594263, 0.3249009233355668, 0.31653387368527885, 0.3253758662771274]
    val_errors = [0.5485003983974457, 0.7469291192835028, 0.5047047184814106, 0.47525712154128336, 0.5542441315000708, 0.4885345152291385, 0.7217185417088595, 0.49489102482795716, 0.5468868069215255, 0.5293076874993065, 0.5160553420673717, 0.4646567440032959, 0.5247112494165247, 0.5063828209313479, 0.49123479528860614, 0.472036325823177, 0.48024770682508294, 0.47736554102464157, 0.46266521670601585, 0.44759295333515514]
    kwargs = {
        'num_epoch': 20,
        'fine_tuned_layers': 10,
        'learning_rate': 0.0001,
        'elapsed_time_sec': 400,
    }
    title = 'Model 7 test'
    html_report = generate_train_report(
        train_errors, val_errors, title, training_parameters=kwargs)
    with open('{}.html'.format(title), mode='w') as report:
        report.write(html_report)


if __name__ == '__main__':
    main()

