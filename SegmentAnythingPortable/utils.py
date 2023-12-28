import numpy


def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = numpy.where(ground_truth_map > 0)
  x_min, x_max = numpy.min(x_indices), numpy.max(x_indices)
  y_min, y_max = numpy.min(y_indices), numpy.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - numpy.random.randint(0, 20))
  x_max = min(W, x_max + numpy.random.randint(0, 20))
  y_min = max(0, y_min - numpy.random.randint(0, 20))
  y_max = min(H, y_max + numpy.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


# help printing a table of report, in logger.
def CreatePrintableTable(rows, headers):
    strRows = []
    for row in rows:
        strRow = []
        for value in row:
            if isinstance(value, (six.string_types, numpy.string_)):
                strRow.append(value)
            elif isinstance(value, dict):
                strRow.append(', '.join(['%s=%r' % (aKey, aVal) for aKey, aVal in value.items()]))
            elif isinstance(value, (list, numpy.ndarray)):
                if len(value) and isinstance(value[0], six.string_types):
                    strRow.append('[' + ', '.join([subValue for subValue in value]) + ']')
                else:
                    strRow.append('[' + ', '.join([(' ' if subValue >= 0 else '-') + '%.3f' % abs(subValue) for subValue in value]) + ']')
            elif isinstance(value, float):
                strRow.append('%0.4f' % value)
            else:
                strRow.append(str(value))
        strRows.append(strRow)

    maxLengths = numpy.array([[len(str(value)) for value in row] for row in strRows]).max(axis=0) + 3
    maxHeaderLength = (max([len(header) for header in headers]) + 3)

    thickSeparator = ' ' + ''.join(['='] * (maxLengths.sum() + len(maxLengths) + maxHeaderLength))
    verticalSeparator = ' ' if len(headers) else '|'
    tableLines = ['\n', thickSeparator]
    tableWidth = (maxLengths.sum() + len(maxLengths) + maxHeaderLength)
    for idxh, row in enumerate(strRows):
        if len(row) == 1 and row[0] in ['===', '...', '---'] and headers[idxh] in ['===', '...', '---']:  # special horizontal separator
            tableLines.append('|' + ''.join([row[0][0]] * tableWidth) + '|')
        else:
            headerText = headers[idxh].ljust(maxHeaderLength) if len(headers) else ''
            tableLines.append('|' + headerText + '|' + verticalSeparator.join([value.ljust(maxLengths[idx]) for idx, value in enumerate(row)]) + '|')
    tableLines.append(thickSeparator)
    return '\n'.join(tableLines)
