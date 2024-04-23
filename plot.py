# # import numpy as np

# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt

# # import collections
# # import time
# # import pickle

# # _since_beginning = collections.defaultdict(lambda: {})
# # _since_last_flush = collections.defaultdict(lambda: {})

# # _iter = [0]
# # def tick():
# # 	_iter[0] += 1

# # def plot(name, value):
# # 	_since_last_flush[name][_iter[0]] = value

# # def flush():
# #     prints = []
# #     for name, vals in _since_last_flush.items():
# #         # Convert the dict_values to a list before calculating the mean
# #         values_list = list(vals.values())
# #         prints.append("{}\\t{}".format(name, np.mean(values_list)))

# #         _since_beginning[name].update(vals)

# #         # Check if _since_beginning[name] is not empty
# #         if _since_beginning[name]:
# #             try:
# #                 x_vals = np.sort(_since_beginning[name].keys())
# #                 y_vals = [_since_beginning[name][x] for x in x_vals]

# #                 plt.clf()
# #                 plt.plot(x_vals, y_vals)
# #                 plt.xlabel('iteration')
# #                 plt.ylabel(name)
# #                 plt.savefig(name.replace(' ', '_')+'.jpg')
# #             except ValueError:
# #                 # Handle empty dictionary gracefully
# #                 pass

# #     print("iter {}\\t{}".format(_iter[0], "\\t".join(prints)))
# #     _since_last_flush.clear()

# #     with open('log.pkl', 'wb') as f:
# #         pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)


# import matplotlib
# matplotlib.use('cairo')  # Set the non-interactive backend before importing pyplot
# import matplotlib.pyplot as plt
# # plt.ioff()

# import numpy as np
# import collections
# import time
# import pickle
# import os

# _since_beginning = collections.defaultdict(lambda: {})
# _since_last_flush = collections.defaultdict(lambda: {})
# _iter = [0]

# # print(matplotlib.get_backend())
# # if 'cairo' in matplotlib.rcsetup.all_backends:
# #     print("The 'cairo' backend is available.")
# # else:
# #     print("The 'cairo' backend is not available.")

# # plot_dir = '/Users/rosshhun/Downloads/DeepSense/images/'
# # if not os.path.exists(plot_dir):
# #     print(f"The directory '{plot_dir}' does not exist.")
# # else:
# #     print(f"The directory '{plot_dir}' exists.")

# def tick():
#     _iter[0] += 1

# def plot(name, value):
#     print(f"Plotting {name} with value {value}")
#     _since_last_flush[name][_iter[0]] = value


# PLOT_DIR = '/Users/rosshhun/Downloads/DeepSense/images/'
# os.makedirs(PLOT_DIR, exist_ok=True)  

# # def flush():
# #     print("Entering flush function")
# #     prints = []
# #     for name, vals in _since_last_flush.items():
# #         # Convert the dict_values to a list before calculating the mean
# #         values_list = list(vals.values())
# #         prints.append("{}\\t{}".format(name, np.mean(values_list)))

# #         _since_beginning[name].update(vals)

# #         # Check if _since_beginning[name] is not empty
# #         if _since_beginning[name]:
# #             try:
# #                 x_vals = np.sort(_since_beginning[name].keys())
# #                 y_vals = [_since_beginning[name][x] for x in x_vals]

# #                 print(f"Plot '{name}':")
# #                 print(f"  x_vals (length {len(x_vals)}): {x_vals}")
# #                 print(f"  y_vals (length {len(y_vals)}): {y_vals}")

# #                 plt.clf()
# #                 plt.plot(x_vals, y_vals)
# #                 plt.xlabel('iteration')
# #                 plt.ylabel(name)
# #                 # plt.savefig(os.path.join(PLOT_DIR, name.replace(' ', '_')+'.jpg'))
# #                 plt.savefig(os.path.join(PLOT_DIR, f"{name.replace(' ', '_')}_{_iter[0]}.jpg"))  # Add iteration number to filename
# #             except Exception as e:
# #                 print(f"Error saving plot '{name}': {e}")
# #                 # Handle empty dictionary gracefully
# #                 # pass
# #         else:
# #             print(f"WARNING: No data found for plot '{name}'")  # Indicate empty data

# #     print("iter {}\\t{}".format(_iter[0], "\\t".join(prints)))
# #     _since_last_flush.clear()

# #     with open('log.pkl', 'wb') as f:
# #         pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)

# def flush():
#     print("Entering flush function")
#     prints = []
#     for name, vals in _since_last_flush.items():
#         print(f"Processing plot '{name}'")
#         values_list = list(vals.values())
#         prints.append("{}\\t{}".format(name, np.mean(values_list)))
#         _since_beginning[name].update(vals)

#         if _since_beginning[name]:
#             print(f"Contents of _since_beginning['{name}']:")
#             print(_since_beginning[name])
#             try:
#                 print("Attempting to create plot...")
#                 print(f"Keys in _since_beginning['{name}']: {list(_since_beginning[name].keys())}")
#                 x_vals = np.sort(_since_beginning[name].keys())
#                 print(f"x_vals before creating y_vals: {x_vals}")
#                 y_vals = [_since_beginning[name][x] for x in x_vals]
#                 print(f"x_vals shape: {np.array(x_vals).shape}")
#                 print(f"y_vals shape: {np.array(y_vals).shape}")
#                 print(f"x_vals: {x_vals}")
#                 print(f"y_vals: {y_vals}")
#                 print(f"x_vals data type: {np.array(x_vals).dtype}")
#                 print(f"y_vals data type: {np.array(y_vals).dtype}")
#                 if len(x_vals) == 0 or len(y_vals) == 0:
#                     print(f"Warning: No data found for plot '{name}'")
#                 else:
#                     plt.clf()
#                     plt.plot(np.array(x_vals), np.array(y_vals))
#                     plt.xlabel('iteration')
#                     plt.ylabel(name)
#                     plt.savefig(os.path.join(PLOT_DIR, f"{name.replace(' ', '_')}_{_iter[0]}.jpg"))
#             except Exception as e:
#                 print(f"Error saving plot '{name}': {e}")
#         else:
#             print(f"WARNING: _since_beginning['{name}'] is empty")

#     print("iter {}\\t{}".format(_iter[0], "\\t".join(prints)))
#     _since_last_flush.clear()

#     with open('log.pkl', 'wb') as f:
#         pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)


# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import collections
# import time
# import pickle
# import os

# _since_beginning = collections.defaultdict(lambda: {})
# _since_last_flush = collections.defaultdict(lambda: {})
# _iter = [0]

# def tick():
#     _iter[0] += 1

# def plot(name, value):
#     print(f"Plotting {name} with value {value}")
#     _since_last_flush[name][_iter[0]] = value


# PLOT_DIR = '/Users/rosshhun/Downloads/DeepSense/images/'
# os.makedirs(PLOT_DIR, exist_ok=True)  


# def flush():
#     print("Entering flush function")
#     prints = []
#     for name, vals in _since_last_flush.items():
#         print(f"Processing plot '{name}'")
#         values_list = list(vals.values())
#         prints.append("{}\\t{}".format(name, np.mean(values_list)))
#         _since_beginning[name].update(vals)

#         if _since_beginning[name]:
#             print(f"Contents of _since_beginning['{name}']:")
#             print(_since_beginning[name])
#             try:
#                 print("Attempting to create plot...")
#                 print(f"Keys in _since_beginning['{name}']: {list(_since_beginning[name].keys())}")
#                 x_vals = np.sort(_since_beginning[name].keys())
#                 y_vals = [_since_beginning[name][x] for x in x_vals]
#                 print(f"x_vals shape: {np.array(x_vals).shape}")
#                 print(f"y_vals shape: {np.array(y_vals).shape}")
#                 print(f"x_vals: {x_vals}")
#                 print(f"y_vals: {y_vals}")
#                 print(f"x_vals data type: {np.array(x_vals).dtype}")
#                 print(f"y_vals data type: {np.array(y_vals).dtype}")
#                 if len(x_vals) == 0 or len(y_vals) == 0:
#                     print(f"Warning: No data found for plot '{name}'")
#                 else:
#                     plt.figure()
#                     sns.lineplot(x=x_vals, y=y_vals)
#                     plt.xlabel('iteration')
#                     plt.ylabel(name)
#                     plt.savefig(os.path.join(PLOT_DIR, f"{name.replace(' ', '_')}_{_iter[0]}.png"))
#                     plt.close()
#             except Exception as e:
#                 print(f"Error saving plot '{name}': {e}")
#         else:
#             print(f"WARNING: _since_beginning['{name}'] is empty")

#     print("iter {}\\t{}".format(_iter[0], "\\t".join(prints)))
#     _since_last_flush.clear()

#     with open('log.pkl', 'wb') as f:
#         pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import collections
import time
import pickle
import os

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_iter = [0]

def tick():
    _iter[0] += 1

def plot(name, value):
    print(f"Plotting {name} with value {value}")
    _since_last_flush[name][_iter[0]] = value

PLOT_DIR = '/Users/rosshhun/Downloads/DeepSense/images/'
os.makedirs(PLOT_DIR, exist_ok=True)

def flush():
    print("Entering flush function")
    prints = []

    # Open a file to save the plot data
    with open('plot_data.txt', 'a') as f:
        for name, vals in _since_last_flush.items():
            print(f"Processing plot '{name}'")
            values_list = list(vals.values())
            prints.append("{}\\t{}".format(name, np.mean(values_list)))
            _since_beginning[name].update(vals)

            if _since_beginning[name]:
                print(f"Contents of _since_beginning['{name}']:")
                print(_since_beginning[name])

                try:
                    print("Attempting to create plot...")
                    print(f"Keys in _since_beginning['{name}']: {list(_since_beginning[name].keys())}")
                    x_vals = np.sort(list(_since_beginning[name].keys()))
                    y_vals = [_since_beginning[name][x] for x in x_vals]

                    print(f"x_vals shape: {np.array(x_vals).shape}")
                    print(f"y_vals shape: {np.array(y_vals).shape}")
                    print(f"x_vals: {x_vals}")
                    print(f"y_vals: {y_vals}")
                    print(f"x_vals data type: {np.array(x_vals).dtype}")
                    print(f"y_vals data type: {np.array(y_vals).dtype}")

                    if len(x_vals) == 0 or len(y_vals) == 0:
                        print(f"Warning: No data found for plot '{name}'")
                    else:
                        plt.figure()
                        sns.lineplot(x=x_vals, y=y_vals)
                        plt.xlabel('iteration')
                        plt.ylabel(name)
                        plt.savefig(os.path.join(PLOT_DIR, f"{name.replace(' ', '_')}_{_iter[0]}.png"))
                        plt.close()

                        # Save the plot data to the file
                        f.write(f"Plot '{name}' - x_vals: {x_vals}, y_vals: {y_vals}\n")

                except Exception as e:
                    print(f"Error saving plot '{name}': {e}")

            else:
                print(f"WARNING: _since_beginning['{name}'] is empty")

    print("iter {}\\t{}".format(_iter[0], "\\t".join(prints)))
    _since_last_flush.clear()

    with open('log.pkl', 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)