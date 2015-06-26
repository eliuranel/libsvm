#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <string.h>
#include <math.h> //fabs
#include "svm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Calloc(type,n) (type *)calloc(n,sizeof(type)) //same as malloc but set allocated memory to 0

struct feature_quant {
	double mean;
	double standard_deviation;
	double feature_min;
	double feature_max;
};

struct feature_ord {
	int min;
	int max;
	int* ties;
	double* ranks;
	int range;
};

union feature {
	struct feature_quant f_quant;
	struct feature_ord f_ord;
};

void exit_with_help()
{
	printf(
	"Usage: svm-scale [options] data_filename\n"
	"options:\n"
	"-y y_lower y_upper : y scaling limits (default: no y scaling)\n"
	"-n : normalize quantitative variables with standardization (default: no standardization)\n"
	"-s save_filename : save scaling parameters to save_filename\n"
	"-r restore_filename : restore scaling parameters from restore_filename\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

char *line = NULL;
int max_line_len = 1024;
double lower=-1.0,upper=1.0;
double y_lower,y_upper;
int y_scaling = 0;
double *feature_max;
double *feature_min;
double y_max = -DBL_MAX;
double y_min = DBL_MAX;
int max_index;
int min_index;
long int num_nonzeros = 0;
long int new_num_nonzeros = 0;
int *data_types;
int standardization = 0;
feature *my_features;

#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))

void output_target(double value);
void output(int index, double value);
void output_quant(int index, double value);
void output_ord(int index, double value);
char* readline(FILE *input);
int clean_up(FILE *fp_restore, FILE *fp, const char *msg);
void construct_data_types(FILE *fp);


int main(int argc,char **argv)
{
	int i, j, index, num_line;
	FILE *fp, *fp_restore = NULL;
	char *save_filename = NULL;
	char *restore_filename = NULL;

	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1]) //Rajouter une option pour la standardization ?
		{
			case 'y':
				y_lower = atof(argv[i]);
				++i;
				y_upper = atof(argv[i]);
				y_scaling = 1;
				break;
			case 'n' : standardization = 1;
			case 's': save_filename = argv[i]; break;
			case 'r': restore_filename = argv[i]; break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}

	if(restore_filename && save_filename)
	{
		fprintf(stderr,"cannot use -r and -s simultaneously\n");
		exit(1);
	}
	
	if(restore_filename && standardization)
	{
		fprintf(stderr,"cannot use -r and -n simultaneously\n");
		exit(1);
	}

	if(argc != i+1) 
		exit_with_help();

	fp=fopen(argv[i],"r");
	
	if(fp==NULL)
	{
		fprintf(stderr,"can't open file %s\n", argv[i]);
		exit(1);
	}

	line = (char *) malloc(max_line_len*sizeof(char));

#define SKIP_TARGET\
	while(isspace(*p)) ++p;\
	while(!isspace(*p)) ++p;

#define SKIP_ELEMENT\
	while(*p!=':') ++p;\
	++p;\
	while(isspace(*p)) ++p;\
	while(*p && !isspace(*p)) ++p;
	
	construct_data_types(fp);
	
	if (data_types[0] == 0)
	{
		/* assumption: min index of attributes is 1 */
		/* pass 1: find out max index of attributes */
		max_index = 0;
		min_index = 1;

		if(restore_filename)
		{
			int idx, c;

			fp_restore = fopen(restore_filename,"r");
			if(fp_restore==NULL)
			{
				fprintf(stderr,"can't open file %s\n", restore_filename);
				exit(1);
			}

			c = fgetc(fp_restore);
			if(c == 'y')
			{
				readline(fp_restore);
				readline(fp_restore);
				readline(fp_restore);
			}
			readline(fp_restore);
			readline(fp_restore);

			while(fscanf(fp_restore,"%d %*f %*f\n",&idx) == 1)
				max_index = max(idx,max_index);
			rewind(fp_restore);
		}

		while(readline(fp)!=NULL)
		{
			char *p=line;

			SKIP_TARGET

			while(sscanf(p,"%d:%*f",&index)==1)
			{
				max_index = max(max_index, index);
				min_index = min(min_index, index);
				SKIP_ELEMENT
				num_nonzeros++;
			}
		}

		if(min_index < 1)
			fprintf(stderr,
				"WARNING: minimal feature index is %d, but indices should start from 1\n", min_index);

		rewind(fp);
	}
	
	if(data_types[0] == 0)
	{
		feature_max = (double *)malloc((max_index+1)* sizeof(double));
		feature_min = (double *)malloc((max_index+1)* sizeof(double));

		if(feature_max == NULL || feature_min == NULL)
		{
			fprintf(stderr,"can't allocate enough memory\n");
			exit(1);
		}

		for(i=0;i<=max_index;i++)
		{
			feature_max[i]=-DBL_MAX;
			feature_min[i]=DBL_MAX;
		}

		/* pass 2: find out min/max value */
		while(readline(fp)!=NULL)
		{
			char *p=line;
			int next_index=1;
			double target;
			double value;

			if (sscanf(p,"%lf",&target) != 1)
				return clean_up(fp_restore, fp, "ERROR: failed to read labels\n");
			y_max = max(y_max,target);
			y_min = min(y_min,target);
			
			SKIP_TARGET

			while(sscanf(p,"%d:%lf",&index,&value)==2)
			{
				for(i=next_index;i<index;i++)
				{
					feature_max[i]=max(feature_max[i],0);
					feature_min[i]=min(feature_min[i],0);
				}
				
				feature_max[index]=max(feature_max[index],value);
				feature_min[index]=min(feature_min[index],value);

				SKIP_ELEMENT
				next_index=index+1;
			}		

			for(i=next_index;i<=max_index;i++)
			{
				feature_max[i]=max(feature_max[i],0);
				feature_min[i]=min(feature_min[i],0);
			}	
		}

		rewind(fp);

		/* pass 2.5: save/restore feature_min/feature_max */
		
		if(restore_filename)
		{
			/* fp_restore rewinded in finding max_index */
			int idx, c;
			double fmin, fmax;
			int next_index = 1;
			
			if((c = fgetc(fp_restore)) == 'y')
			{
				if(fscanf(fp_restore, "%lf %lf\n", &y_lower, &y_upper) != 2 ||
				  fscanf(fp_restore, "%lf %lf\n", &y_min, &y_max) != 2)
					return clean_up(fp_restore, fp, "ERROR: failed to read scaling parameters\n");
				y_scaling = 1;
			}
			else
				ungetc(c, fp_restore);

			if (fgetc(fp_restore) == 'x') 
			{
				if(fscanf(fp_restore, "%lf %lf\n", &lower, &upper) != 2)
					return clean_up(fp_restore, fp, "ERROR: failed to read scaling parameters\n");
				while(fscanf(fp_restore,"%d %lf %lf\n",&idx,&fmin,&fmax)==3)
				{
					for(i = next_index;i<idx;i++)
						if(feature_min[i] != feature_max[i])
							fprintf(stderr,
								"WARNING: feature index %d appeared in file %s was not seen in the scaling factor file %s.\n",
								i, argv[argc-1], restore_filename);

					feature_min[idx] = fmin;
					feature_max[idx] = fmax;

					next_index = idx + 1;
				}
				
				for(i=next_index;i<=max_index;i++)
					if(feature_min[i] != feature_max[i])
						fprintf(stderr,
							"WARNING: feature index %d appeared in file %s was not seen in the scaling factor file %s.\n",
							i, argv[argc-1], restore_filename);
			}
			fclose(fp_restore);
		}

		if(save_filename)
		{
			FILE *fp_save = fopen(save_filename,"w");
			if(fp_save==NULL)
			{
				fprintf(stderr,"can't open file %s\n", save_filename);
				exit(1);
			}
			if(y_scaling)
			{
				fprintf(fp_save, "y\n");
				fprintf(fp_save, "%.16g %.16g\n", y_lower, y_upper);
				fprintf(fp_save, "%.16g %.16g\n", y_min, y_max);
			}
			fprintf(fp_save, "x\n");
			fprintf(fp_save, "%.16g %.16g\n", lower, upper);
			for(i=1;i<=max_index;i++)
			{
				if(feature_min[i]!=feature_max[i])
					fprintf(fp_save,"%d %.16g %.16g\n",i,feature_min[i],feature_max[i]);
			}

			if(min_index < 1)
				fprintf(stderr,
					"WARNING: scaling factors with indices smaller than 1 are not stored to the file %s.\n", save_filename);

			fclose(fp_save);
		}
		
		/* pass 3: scale */
		while(readline(fp)!=NULL)
		{
			char *p=line;
			int next_index=1;
			double target;
			double value;
			
			if (sscanf(p,"%lf",&target) != 1)
				return clean_up(NULL, fp, "ERROR: failed to read labels\n");
			output_target(target);

			SKIP_TARGET

			while(sscanf(p,"%d:%lf",&index,&value)==2)
			{
				for(i=next_index;i<index;i++)
					output(i,0);
				
				output(index,value);

				SKIP_ELEMENT
				next_index=index+1;
			}		

			for(i=next_index;i<=max_index;i++)
				output(i,0);

			printf("\n");
		}

		if (new_num_nonzeros > num_nonzeros)
			fprintf(stderr, 
				"WARNING: original #nonzeros %ld\n"
				"         new      #nonzeros %ld\n"
				"Use -l 0 if many original feature values are zeros\n",
				num_nonzeros, new_num_nonzeros);
	} else {
		
		my_features = Malloc(feature,max_index);
		
		if(restore_filename) //Retrieve features from restore_filename
		{
			int idx, c;
			int int_value1, int_value2, int_value3;
			double dble_value1, dble_value2, dble_value3, dble_value4;
			char *val;
			char *endptr;
			
			fp_restore = fopen(restore_filename,"r");
			if(fp_restore==NULL)
			{
				fprintf(stderr,"can't open file %s\n", restore_filename);
				exit(1);
			}
			
			if((c = fgetc(fp_restore)) == 'y')
			{
				if(fscanf(fp_restore, "%lf %lf\n", &y_lower, &y_upper) != 2 ||
				  fscanf(fp_restore, "%lf %lf\n", &y_min, &y_max) != 2)
					return clean_up(fp_restore, fp, "ERROR: failed to read scaling parameters\n");
				y_scaling = 1;
			} else {
				ungetc(c, fp_restore);
			}
			
			if (fgetc(fp_restore) == 'x') 
			{
				if (fscanf(fp_restore, "%d\n", &standardization) !=1)
					return clean_up(fp_restore, fp, "ERROR: failed to read scaling parameters\n");
				while(fscanf(fp_restore, "%d: ", &idx) ==1)
				{
					switch(data_types[idx]){
						case QUANT:
							if (fscanf(fp_restore, "%lf %lf %lf %lf\n", &dble_value1, &dble_value2, &dble_value3, &dble_value4) !=4)
							{
								return clean_up(fp_restore, fp, "ERROR: failed to read scaling parameters\n");
							} else {
								(my_features[idx].f_quant).mean = dble_value1;
								(my_features[idx].f_quant).standard_deviation = dble_value2;
								(my_features[idx].f_quant).feature_min = dble_value3;
								(my_features[idx].f_quant).feature_max = dble_value4;
							}
							break;
						case ORD:
							if (fscanf(fp_restore, "%d %d %d\n", &int_value1, &int_value2, &int_value3) !=3)
							{
								return clean_up(fp_restore, fp, "ERROR: failed to read scaling parameters\n");
							} else {
								(my_features[idx].f_ord).min = int_value1;
								(my_features[idx].f_ord).max = int_value2;
								(my_features[idx].f_ord).range = int_value3;

								(my_features[idx].f_ord).ranks = Malloc(double, (my_features[idx].f_ord).max - (my_features[idx].f_ord).min +1);
								
								if (readline(fp_restore) != NULL){
									val = strtok(line, " \t\n");
									for (i = (my_features[idx].f_ord).min; i < (my_features[idx].f_ord).max; i++)
									{
										//errno = 0
										(my_features[idx].f_ord).ranks[i] = strtod(val,&endptr);
										//TODO : if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))) -> error
										val = strtok(NULL, " \t\n");
									}
									(my_features[idx].f_ord).ranks[i] = strtod(val,&endptr);
								} //else error ?
							}
							break;
						default:
							readline(fp_restore);
							break;
					}
				}
			}
			
			fclose(fp_restore);
			
		} else { //Compute features from input
			//PASS 1
			//    QUANT: find out the mean + the min and max
			//    ORD: find out the min and max
		  
			//PASS 1.5
			//    ORD: calloc ties and ranks
		  
			//PASS 2
			//    QUANT: find out the standard deviation
			//    ORD: compute the ties
		  
			//PASS 2.5 (= no parsing)
			//    QUANT: compute feature_min and feature_max
			//    ORD: compute the ranks and the range
		  
		}
	  
		if(save_filename) //Save features in save_filename
		{
			FILE *fp_save = fopen(save_filename,"w");
			if(fp_save==NULL)
			{
				fprintf(stderr,"can't open file %s\n", save_filename);
				exit(1);
			}
			if(y_scaling)
			{
				fprintf(fp_save, "y\n");
				fprintf(fp_save, "%.16g %.16g\n", y_lower, y_upper);
				fprintf(fp_save, "%.16g %.16g\n", y_min, y_max);
			}
			fprintf(fp_save, "x\n");
			fprintf(fp_save, "%d\n", standardization);
			for(i=1;i<=max_index;i++)
			{
				switch(data_types[i])
				{
					case QUANT:
						fprintf(fp_save, "%d: %.16g %.16g %.16g %.16g\n", i, (my_features[i].f_quant).mean, (my_features[i].f_quant).standard_deviation, (my_features[i].f_quant).feature_min, (my_features[i].f_quant).feature_max);
						break;
					case ORD:
						fprintf(fp_save, "%d: %d %d %d\n", i, (my_features[i].f_ord).min, (my_features[i].f_ord).max, (my_features[i].f_ord).range);
						for (j = (my_features[i].f_ord).min; j <= (my_features[i].f_ord).max; j++)
						{
							fprintf(fp_save, "%.16g ", (my_features[i].f_ord).ranks[j - (my_features[i].f_ord).min]);
						}
						break;
				}
			}
			if(min_index < 1)
				fprintf(stderr,
					"WARNING: scaling factors with indices smaller than 1 are not stored to the file %s.\n", save_filename);
			fclose(fp_save);
		}
		
		// Write new values
		for(i=0; i<data_types[0]; i++)
		{
			if(readline(fp) != NULL)
				printf(line);
		}
		
		num_line = data_types[0];
		
		while(readline(fp)!=NULL)
		{
			char *p=line;
			int next_index=1;
			double target;
			double dble_value, dble_value1, dble_value2, dble_value3, dble_value4;
			int int_value, int_value1, int_value2;
			char *string_value;
			char char_value;
			
			num_line++;
	  
			if (sscanf(p,"%lf",&target) != 1)
				return clean_up(NULL, fp, "ERROR: failed to read labels\n");
			output_target(target);

			SKIP_TARGET

			while(sscanf(p,"%d:",&index)==1)
			{
				for(i=next_index;i<index;i++)
					output(i,0);
				
				switch(data_types[index])
				{
					case QUANT:
						if(sscanf(p,"%lf",&dble_value)!=1) {
							exit_input_error(num_line);
						} else {
							output_quant(index,dble_value);
						}
						break;
					case ORD:
						if(sscanf(p,"%lf",&dble_value)!=1) {
							exit_input_error(num_line);
						} else {
							output_ord(index,dble_value);
						}
						break;
					case DICH:
						if(sscanf(p,"%c",&char_value)!=1) {
							exit_input_error(num_line);
						} else {
							printf("%d:%c ",index,char_value);
						}
					case C_CIRC:
						if(sscanf(p,"%lf",&dble_value)!=1) {
							exit_input_error(num_line);
						} else {
							printf("%d:%g ",index,dble_value);
						}
						break;  
					case D_CIRC:
						if(sscanf(p,"%d,%d",&int_value1,&int_value2)!=2) {
							exit_input_error(num_line);
						} else {
							printf("%d:%d,%d ",index,int_value1,int_value2);
						}
						break;
					case FUZZ:
						if(sscanf(p,"%lf,%lf,%lf,%lf",&dble_value1,&dble_value2,&dble_value3,&dble_value4)!=4) {
							exit_input_error(num_line);
						} else {
							printf("%lf,%lf,%lf,%lf",dble_value1,dble_value2,dble_value3,dble_value4);
						}
						break;
					case MULT:
						if(sscanf(p,"%d",&int_value)!=1) {
							exit_input_error(num_line);
						} else {
							printf("%d:%d ",index,int_value);
						}
						break;
					case NOM:
						if(sscanf(p,"%s",string_value)!=1) { //TODO : OK ???
							exit_input_error(num_line);
						} else {
							printf("%d:%s ",index,string_value);
						}
						break; 
				}
					
				SKIP_ELEMENT
				next_index=index+1;
			}		

			for(i=next_index;i<=max_index;i++)
				output(i,0);

			printf("\n");
		}
	}

	free(line);
	free(feature_max);
	free(feature_min);
	fclose(fp);
	return 0;
}

char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void output_target(double value)
{
	if(y_scaling)
	{
		if(value == y_min)
			value = y_lower;
		else if(value == y_max)
			value = y_upper;
		else value = y_lower + (y_upper-y_lower) *
			     (value - y_min)/(y_max-y_min);
	}
	printf("%g ",value);
}

void output(int index, double value)
{
	/* skip single-valued attribute */
	if(feature_max[index] == feature_min[index])
		return;

	if(value == feature_min[index])
		value = lower;
	else if(value == feature_max[index])
		value = upper;
	else
		value = lower + (upper-lower) * 
			(value-feature_min[index])/
			(feature_max[index]-feature_min[index]);

	if(value != 0)
	{
		printf("%d:%g ",index, value);
		new_num_nonzeros++;
	}
}

void output_quant(int index, double value)
{
	if (standardization)
	{
		value = value - (my_features[index-1].f_quant).mean;
		value = value / (my_features[index-1].f_quant).standard_deviation;
	}
	value = value / fabs((my_features[index-1].f_quant).feature_max - (my_features[index-1].f_quant).feature_min);
	printf("%d:%g ",index, value);
}

void output_ord(int index, double value)
{
	int i;
	if (value < (my_features[index-1].f_ord).min)
		value = 0;
	if (value > (my_features[index-1].f_ord).max)
		value = (my_features[index-1].f_ord).range;
	if (value >= (my_features[index-1].f_ord).min && value <= (my_features[index-1].f_ord).max)
	{
		if ((my_features[index-1].f_ord).ranks[(int)value] == -1){
			for (i = (int)value; i <= (my_features[index-1].f_ord).max; i++){
				if ((my_features[index-1].f_ord).ranks[i] != -1)
				{
					value = (my_features[index-1].f_ord).ranks[i];
					break;
				}
			}
			for (i = (int)value; i >= (my_features[index-1].f_ord).min; i--){
				if ((my_features[index-1].f_ord).ranks[i] != -1)
				{
					value = value + (my_features[index-1].f_ord).ranks[i];
					break;
				}
			}
			value = value / 2.0;
		} else {
			value = (my_features[index-1].f_ord).ranks[(int)value];
		}
	}
	value = value / (my_features[index-1].f_ord).range;
	printf("%d:%g ",index, value);
}

int clean_up(FILE *fp_restore, FILE *fp, const char* msg)
{
	fprintf(stderr,	"%s", msg);
	free(line);
	free(feature_max);
	free(feature_min);
	fclose(fp);
	if (fp_restore)
		fclose(fp_restore);
	return -1;
}

void construct_data_types(FILE *fp)
{
	int num_line, i_min, i_max, i;
	char *endptr;
	char *label, *idx_min, *idx_max;
	if(readline(fp)!=NULL)
	{
		char *pp = strtok(line," \t");
		if(strcmp(pp,"TYPES")==0)
		{
			label = strtok(NULL," \t\n");
			if(label == NULL)
				exit_input_error(1);
			max_index = (int) strtol(label,&endptr,10);  //TODO qu'est-ce que le 10 ??
			if(endptr == label || *endptr != '\0')
				exit_input_error(1);
			data_types = Calloc(int, max_index+1);
			data_types[0] = 1;
		} else {
			data_types = Calloc(int,1);
			rewind(fp);
		}
	}
	
	num_line = 1;
	while(readline(fp)!=NULL)
	{
		num_line++;
		char *pp = strtok(line," \t");
		if(strcmp(pp,"TYPES")==0)
		{
			data_types[0]++;
			label = strtok(NULL," \t\n");
			while(1){
				idx_min = strtok(NULL,">");
				if(idx_min == NULL)
					break;
				idx_max = strtok(NULL,", \t\n");
				if(idx_max == NULL)
					break;
				errno = 0;
				i_min = (int) strtol(idx_min,&endptr,10);  //TODO qu'est-ce que le 10 ??
				if(endptr == idx_min || errno != 0 || *endptr != '\0')
					exit_input_error(num_line+1);
				i_max = (int) strtol(idx_max,&endptr,10);  //TODO qu'est-ce que le 10 ??
				if(endptr == idx_max || errno != 0 || *endptr != '\0')
					exit_input_error(num_line+1);
				for (i = i_min; i <= i_max; i++)
				{
					data_types[i] = Types_to_int(label);
				}
			}
		}
	}
}
