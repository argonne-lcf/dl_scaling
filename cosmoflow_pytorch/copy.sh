#!/bin/bash
for file in r*.sc
do
  cp "$file" "${file%.sc}_hvd.sc"
done

